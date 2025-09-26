#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多任务降水校正训练脚本（增强版）

新增/修改要点（本版本）:
1. 痕量归零：观测 <0.1 mm 直接置 0，减少“假湿”。
2. 发生概率与雨量彻底分离：评估/最终输出时先按概率阈值裁剪，再截断小毛毛雨。
3. 分位数损失逐样本加权：按雨量等级给不同权重（中庸方案）。
4. 可调平衡系数：lambda_occ / lambda_q 控制分类与回归的相对力度。
5. 评估函数支持：occ_prob_threshold_eval + drizzle_cut（小雨截断）+ tail 平滑放大。
6. tail 调整放在所有过滤之后，避免误放大小雨。
7. 保留 WeightedRandomSampler（可选），默认不使用。
8. 结构与原逻辑兼容，方便继续扩展（加入更多特征、极值建模等）。

建议使用步骤：
  第一步：不开 tail，主要验证“假湿下降 + 中雨不再压扁”。
  第二步：观察大雨仍低估时，再开启 --enable_tail_adjust 看尾部改善幅度。
"""

import os
import re
import glob
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

# =========================================================
# 命令行参数
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_dir', type=str, default='./output_multitask')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patch_size', type=int, default=11)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
parser.add_argument('--use_weighted_sampler', action='store_true', help='是否用加权采样提升中高雨频率')
parser.add_argument('--enable_tail_adjust', action='store_true', help='训练后输出评估时启用尾部乘法平滑修正')
parser.add_argument('--tail_threshold', type=float, default=25.0, help='尾部修正起始阈值(mm)')
parser.add_argument('--tail_smooth_width', type=float, default=4.0, help='尾部过渡平滑宽度')

# 新增：损失权重与评估控制
parser.add_argument('--lambda_occ', type=float, default=1.5, help='发生概率损失权重 (建议 1.2~2)')
parser.add_argument('--lambda_q', type=float, default=1.0, help='分位损失权重')
parser.add_argument('--occ_prob_threshold_eval', type=float, default=0.5,
                    help='评估/最终统计使用的发生概率阈值 (p < 阈值 -> 预测=0)')
parser.add_argument('--drizzle_cut', type=float, default=0.15,
                    help='评估阶段 0 < 预测 < drizzle_cut 认定为 0 (去除毛毛雨)')
args = parser.parse_args()

# =========================================================
# 随机种子
# =========================================================
def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================
# 常量 / 归一化参数
# =========================================================
ERA5_MAX = 380.15
CRA_MAX = 121.902878
PSURF_MIN, PSURF_MAX = 61681.609609, 101344.341562
QAIR_MIN, QAIR_MAX = 0.0, 0.024567
DEM_MIN, DEM_MAX = 429, 4439

# =========================================================
# 数据路径
# =========================================================
ERA5_DIR = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/ERA5_Daily_mm"
ERA5_VAR = "daily_precip_mm"
CRA_DIR = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/CRA_Pre_Daily"
CRA_VAR = "TotalPrecip_tavg"
PSURF_DIR = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/CRA_Psurf_Daily"
PSURF_VAR = "Psurf_f_inst"
QAIR_DIR = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/CRA_Qair_Daily"
QAIR_VAR = "Qair_f_inst"
DEM_PATH = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/Terrain/DEM.tif"
SLOPE_PATH = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/Terrain/SlopeD.tif"
STATION_CSV = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/StationData_with_coords.csv"
NINO_CSV = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/nina34.anom.csv"

# 训练时间段
TRAIN_START = pd.to_datetime('2006-01-01')
TRAIN_END   = pd.to_datetime('2010-12-31')

# =========================================================
# 工具函数
# =========================================================
def load_nc_date_var_map(nc_dir, varname, time_name=None):
    date_map = {}
    files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
    for f in files:
        ds = xr.open_dataset(f)
        used_time = time_name
        if not used_time:
            if 'time' in ds.coords or 'time' in ds.dims:
                used_time = 'time'
            elif 'date' in ds.coords or 'date' in ds.dims:
                used_time = 'date'
        if used_time and (used_time in ds.coords or used_time in ds.dims):
            times = ds[used_time].values
            for i, t in enumerate(times):
                if np.issubdtype(type(t), np.datetime64):
                    date_str = pd.to_datetime(str(t)).strftime('%Y-%m-%d')
                else:
                    date_str = str(t)
                arr = ds[varname].isel({used_time: i}).values
                date_map[date_str] = arr
        else:
            fname = os.path.basename(f)
            m = re.search(r'(\d{4}-?\d{2}-?\d{2})', fname)
            if m:
                s = m.group(1)
                if '-' in s:
                    date_str = s
                else:
                    date_str = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                arr = ds[varname].values
                if arr.ndim == 2:
                    date_map[date_str] = arr
                elif arr.ndim > 2:
                    date_map[date_str] = arr[0]
            else:
                print(f"[WARN] 无法识别文件名日期: {fname}")
        ds.close()
    return date_map

def load_dem_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        tr = src.transform
    return arr, tr

def find_latlon_idx(transform, lat, lon):
    col, row = ~transform * (lon, lat)
    return int(round(row)), int(round(col))

def load_nino_csv(path):
    df = pd.read_csv(path)
    nino_map = {}
    cols_lower = [c.lower() for c in df.columns]
    if "year" in cols_lower and "month" in cols_lower:
        ycol = df.columns[cols_lower.index("year")]
        mcol = df.columns[cols_lower.index("month")]
        vcol = [c for c in df.columns if c.lower() in ("value", "nino", "anom")][0]
        for _, r in df.iterrows():
            period = pd.Period(year=int(r[ycol]), month=int(r[mcol]), freq='M')
            nino_map[period] = float(r[vcol])
    elif "date" in cols_lower:
        dcol = df.columns[cols_lower.index("date")]
        vcol = [c for c in df.columns if c.lower() in ("value", "nino", "anom")][0]
        for _, r in df.iterrows():
            dt = pd.to_datetime(str(r[dcol]), errors='coerce')
            if pd.isna(dt): continue
            period = pd.Period(year=dt.year, month=dt.month, freq='M')
            nino_map[period] = float(r[vcol])
    else:
        raise ValueError("NINO CSV需有year/month/value或date/value列")
    return nino_map

def grade_precip(value):
    if value < 0.1: return 0
    elif value < 10: return 1
    elif value < 25: return 2
    elif value < 50: return 3
    elif value < 100: return 4
    elif value < 250: return 5
    else: return 6

def get_month_onehot(date):
    m = pd.to_datetime(date).month
    onehot = np.zeros(12, dtype=np.float32)
    onehot[m-1] = 1
    return onehot

# =========================================================
# Dataset
# =========================================================
class PrecipDatasetNC(Dataset):
    def __init__(self,
                 era5_map, cra_map, psurf_map, qair_map,
                 dem_arr, dem_tr, slope_arr,
                 station_df, patch_size=11, nino_map=None,
                 pad_mode='reflect'):
        self.era5_map = era5_map
        self.cra_map = cra_map
        self.psurf_map = psurf_map
        self.qair_map = qair_map
        self.dem_arr = dem_arr
        self.dem_tr = dem_tr
        self.slope_arr = slope_arr
        self.station_df = station_df
        self.patch_size = patch_size
        self.nino_map = nino_map
        self.pad_mode = pad_mode

        self.features = []
        self.targets = []
        self.era5_center_vals = []
        self.era5_center_grade = []
        self.obs_grade = []
        self.meta_rows = []

        print("[INFO] 采样站点 patch ...")
        for _, row in tqdm(station_df.iterrows(), total=len(station_df)):
            date_str = pd.to_datetime(row['date']).strftime("%Y-%m-%d")
            lat, lon = row['y'], row['x']
            if (date_str not in era5_map or date_str not in cra_map or
                date_str not in psurf_map or date_str not in qair_map or
                np.isnan(row['Prcp'])):
                continue

            r_idx, c_idx = find_latlon_idx(self.dem_tr, lat, lon)

            def extract_patch(arr):
                half = self.patch_size // 2
                mode = 'reflect' if self.pad_mode == 'reflect' else 'edge'
                padded = np.pad(arr, pad_width=half, mode=mode)
                r = r_idx + half
                c = c_idx + half
                patch = padded[r-half:r+half+1, c-half:c+half+1]
                if patch.shape != (self.patch_size, self.patch_size):
                    return None
                return patch

            era5_patch  = extract_patch(self.era5_map[date_str])
            cra_patch   = extract_patch(self.cra_map[date_str])
            psurf_patch = extract_patch(self.psurf_map[date_str])
            qair_patch  = extract_patch(self.qair_map[date_str])
            dem_patch   = extract_patch(self.dem_arr)
            slope_patch = extract_patch(self.slope_arr)
            if any(p is None for p in [era5_patch, cra_patch, psurf_patch, qair_patch, dem_patch, slope_patch]):
                continue

            aspect_rad = np.deg2rad(slope_patch)
            aspect_sin = np.sin(aspect_rad)
            aspect_cos = np.cos(aspect_rad)

            era5_grade_patch = np.vectorize(grade_precip)(era5_patch).astype(np.float32) / 6.0

            era5_patch_norm  = np.clip(era5_patch / ERA5_MAX, 0, 1)
            cra_patch_norm   = np.clip(cra_patch / CRA_MAX, 0, 1)
            psurf_patch_norm = np.clip((psurf_patch - PSURF_MIN)/(PSURF_MAX-PSURF_MIN), 0, 1)
            qair_patch_norm  = np.clip((qair_patch - QAIR_MIN)/(QAIR_MAX-QAIR_MIN), 0, 1)
            dem_patch_norm   = np.clip((dem_patch - DEM_MIN)/(DEM_MAX-DEM_MIN), 0, 1)

            month_onehot = get_month_onehot(date_str)
            month_channels = month_onehot[:, None, None] * np.ones((12, self.patch_size, self.patch_size), dtype=np.float32)

            period = pd.to_datetime(date_str).to_period('M')
            nino_value = self.nino_map.get(period, 0.0) if self.nino_map else 0.0
            nino_layer = np.ones((self.patch_size, self.patch_size), dtype=np.float32) * nino_value

            patch = np.concatenate([
                np.stack([
                    era5_patch_norm, cra_patch_norm, dem_patch_norm,
                    aspect_sin, aspect_cos,
                    psurf_patch_norm, qair_patch_norm, era5_grade_patch
                ], axis=0),
                month_channels,
                nino_layer[None, ...]
            ], axis=0)  # (21,H,W)

            if np.any(np.isnan(patch)):
                continue

            obs_precip = float(row['Prcp'])
            # 痕量归零 —— 关键
            if obs_precip < 0.1:
                obs_precip = 0.0

            era5_center = float(era5_patch[self.patch_size//2, self.patch_size//2])
            era5_grade_center = grade_precip(era5_center)
            obs_grade_center = grade_precip(obs_precip)

            self.features.append(patch)
            self.targets.append(obs_precip)
            self.era5_center_vals.append(era5_center)
            self.era5_center_grade.append(era5_grade_center)
            self.obs_grade.append(obs_grade_center)

            self.meta_rows.append({
                "station": row['station'],
                "date": date_str,
                "lat": lat,
                "lon": lon,
                "obs": obs_precip,
                "era5_center": era5_center,
                "obs_grade": obs_grade_center,
                "era5_grade": era5_grade_center,
                "nino": nino_value
            })

        self.features = np.array(self.features, dtype=np.float32)
        self.targets  = np.array(self.targets, dtype=np.float32)
        self.era5_center_vals = np.array(self.era5_center_vals, dtype=np.float32)
        self.era5_center_grade = np.array(self.era5_center_grade, dtype=np.int64)
        self.obs_grade = np.array(self.obs_grade, dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.era5_center_vals[idx], dtype=torch.float32),
            torch.tensor(self.era5_center_grade[idx], dtype=torch.long),
            torch.tensor(self.obs_grade[idx], dtype=torch.long)
        )

    def export_samples_csv(self, out_csv):
        pd.DataFrame(self.meta_rows).to_csv(out_csv, index=False)
        print(f"[INFO] 样本元数据导出: {out_csv}  共 {len(self.meta_rows)} 条")

# =========================================================
# 模型
# =========================================================
class MultiTaskQuantileCNN(nn.Module):
    def __init__(self, in_channels, patch_size, quantiles=(0.5,0.9,0.95)):
        super().__init__()
        self.quantiles = quantiles
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat_dim = 96
        self.occ_head   = nn.Linear(feat_dim, 1)
        self.resid_head = nn.Linear(feat_dim, len(self.quantiles))

    def forward(self, x):
        h = self.backbone(x)
        h = self.gap(h).flatten(1)
        occ_logit = self.occ_head(h).squeeze(1)
        resid_q   = self.resid_head(h)
        return occ_logit, resid_q

# =========================================================
# 分位数损失（逐样本加权版本）
# =========================================================
def weighted_multi_quantile_loss(pred_log_q, y_log, quantiles, weights):
    """
    pred_log_q: [N, Q] (log 空间)
    y_log:      [N]
    weights:    [N]  (雨日样本权重)
    """
    total = 0.0
    wsum = weights.sum() + 1e-6
    for i, q in enumerate(quantiles):
        diff = y_log - pred_log_q[:, i]          # [N]
        pin  = torch.maximum(q * diff, (q - 1) * diff)  # pinball per-sample
        total += (pin * weights).sum() / wsum
    return total / len(quantiles)

# =========================================================
# 强度权重（中庸版数组）
# 等级: 0(<0.1) 1(0.1-10) 2(10-25) 3(25-50) 4(50-100) 5(100-250) 6(>=250)
# =========================================================
def build_intensity_weights(obs_grade_tensor):
    base_weights = torch.tensor([1.0, 1.1, 1.3, 1.7, 2.2, 2.9, 3.4],
                                device=obs_grade_tensor.device)
    return base_weights[obs_grade_tensor]

# =========================================================
# 事件评分 / 分箱 / 尾部平滑
# =========================================================
def event_scores(pred, obs, thresholds=(0.1,1,10,25,50)):
    out = {}
    for thr in thresholds:
        ph = pred >= thr
        oh = obs >= thr
        hits = np.sum(ph & oh)
        miss = np.sum(~ph & oh)
        fal  = np.sum(ph & ~oh)
        pod = hits / (hits + miss + 1e-9)
        far = fal / (hits + fal + 1e-9)
        csi = hits / (hits + miss + fal + 1e-9)
        out[thr] = {"POD": float(pod), "FAR": float(far), "CSI": float(csi),
                    "Hits": int(hits), "Miss": int(miss), "FalseAlarms": int(fal)}
    return out

def bin_stats(pred, obs, bins=(0,1,5,10,20,30,50,100,1e9)):
    idx = np.digitize(obs, bins) - 1
    rec = []
    for i in range(len(bins)-1):
        mask = idx == i
        if not np.any(mask): continue
        p = pred[mask]; o = obs[mask]
        bias = np.mean(p - o)
        rmse = math.sqrt(mean_squared_error(o, p))
        rec.append({
            "bin": f"{bins[i]}-{bins[i+1]}",
            "n": int(mask.sum()),
            "bias": float(bias),
            "rmse": float(rmse),
            "obs_mean": float(o.mean()),
            "pred_mean": float(p.mean())
        })
    return rec

def compute_tail_ratio(era5_vals, obs_vals, threshold=25.0):
    mask = (obs_vals >= threshold) & (era5_vals >= 5)
    if mask.sum() < 5:
        print("[WARN] 高强度样本太少，tail_ratio=1.0")
        return 1.0
    ratios = obs_vals[mask] / np.maximum(era5_vals[mask], 1e-6)
    return float(np.median(ratios))

def smooth_weight(values, threshold, width):
    return 0.5 * (1 + np.tanh((values - threshold)/(width + 1e-6)))

def apply_tail_adjustment(raw_pred, era5_raw, tail_ratio, threshold, width):
    w = smooth_weight(era5_raw, threshold, width)
    extreme_adjust = era5_raw * tail_ratio
    return (1 - w) * raw_pred + w * extreme_adjust

# =========================================================
# 评估（核心：概率阈值 & drizzle 截断 → 尾部平滑）
# =========================================================
def evaluate_model(model, dataloader, device, quantiles=(0.5,0.9,0.95),
                   tail_adjust_cfg=None,
                   occ_prob_threshold=0.5,
                   drizzle_cut=0.15,
                   use_occ_threshold=True):
    model.eval()
    all_obs, all_pred_mid, all_era5 = [], [], []

    with torch.no_grad():
        for X, y, era5_c, _, obs_grade in dataloader:
            X = X.to(device)
            era5_c_dev = era5_c.to(device)

            occ_logit, resid_q = model(X)
            p = torch.sigmoid(occ_logit).cpu().numpy()

            era5_log = torch.log1p(era5_c_dev.clamp(min=0))
            pred_log_q = era5_log.unsqueeze(1) + resid_q
            mid_idx = quantiles.index(0.5)
            pred_mid = torch.expm1(pred_log_q[:, mid_idx]).cpu().numpy()

            # 1) 发生概率阈值
            if use_occ_threshold and occ_prob_threshold > 0:
                pred_mid = np.where(p >= occ_prob_threshold, pred_mid, 0.0)

            # 2) drizzle 截断（去掉小毛毛雨）
            if drizzle_cut > 0:
                pred_mid = np.where((pred_mid > 0) & (pred_mid < drizzle_cut), 0.0, pred_mid)

            # 3) 尾部平滑 (只在已经过滤后的值基础上)
            era5_np = era5_c.numpy()
            if tail_adjust_cfg is not None:
                pred_mid = apply_tail_adjustment(
                    pred_mid, era5_np,
                    tail_adjust_cfg['ratio'],
                    tail_adjust_cfg['threshold'],
                    tail_adjust_cfg['width']
                )

            all_obs.append(y.numpy())
            all_pred_mid.append(pred_mid)
            all_era5.append(era5_np)

    obs  = np.concatenate(all_obs)
    pred = np.concatenate(all_pred_mid)
    era5 = np.concatenate(all_era5)

    m = ~np.isnan(obs) & ~np.isnan(pred)
    obs, pred, era5 = obs[m], pred[m], era5[m]

    bias = float((pred - obs).mean())
    rmse = math.sqrt(mean_squared_error(obs, pred))
    corr = pearsonr(pred, obs)[0] if len(obs) > 2 else float('nan')

    e_bias = float((era5 - obs).mean())
    e_rmse = math.sqrt(mean_squared_error(obs, era5))
    e_corr = pearsonr(era5, obs)[0] if len(obs) > 2 else float('nan')

    bins_res = bin_stats(pred, obs)
    evt_res  = event_scores(pred, obs)

    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    q_table = []
    for q in qs:
        q_table.append({
            "q": q,
            "Obs": float(np.quantile(obs, q)),
            "Pred": float(np.quantile(pred, q)),
            "ERA5": float(np.quantile(era5, q))
        })

    return {
        "Model": {"Bias": bias, "RMSE": rmse, "Corr": corr},
        "ERA5":  {"Bias": e_bias, "RMSE": e_rmse, "Corr": e_corr},
        "Bins": bins_res,
        "Events": evt_res,
        "Quantiles": q_table,
        "N": int(len(obs)),
        "Params": {
            "occ_prob_threshold_eval": occ_prob_threshold,
            "drizzle_cut": drizzle_cut,
            "tail_used": tail_adjust_cfg is not None
        }
    }

# =========================================================
# 训练
# =========================================================
def train(model, train_loader, val_loader, device, epochs, lr,
          quantiles=(0.5,0.9,0.95),
          tail_adjust_cfg=None,
          output_dir='./output',
          lambda_occ=1.5, lambda_q=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_rmse = float('inf')
    best_state = None
    patience = 8
    patience_counter = 0

    for epoch in range(1, epochs+1):
        model.train()
        run_loss = run_occ = run_q = 0.0
        nb = 0

        train_iter = tqdm(train_loader, desc=f"[Epoch {epoch}] Train", leave=False)
        for (X, y, era5_c, _, obs_grade) in train_iter:
            X = X.to(device)
            y = y.to(device)
            era5_c = era5_c.to(device)
            obs_grade = obs_grade.to(device)

            occ_label = (y > 0.1).float()   # 发生标签
            y_log     = torch.log1p(y)
            era5_log  = torch.log1p(era5_c.clamp(min=0))

            occ_logit, resid_q = model(X)
            pred_log_q = era5_log.unsqueeze(1) + resid_q

            # 分类损失
            occ_loss = F.binary_cross_entropy_with_logits(occ_logit, occ_label)

            # 只对有雨样本做分位数
            rain_mask = occ_label > 0
            if rain_mask.any():
                pred_log_q_rain = pred_log_q[rain_mask]
                y_log_rain      = y_log[rain_mask]
                w_rain = build_intensity_weights(obs_grade[rain_mask]).float()
                # （可选：对≥25mm 再乘额外 1.2，这里先不加）
                q_loss = weighted_multi_quantile_loss(pred_log_q_rain, y_log_rain, quantiles, w_rain)
            else:
                q_loss = torch.zeros([], device=device)

            total_loss = lambda_occ * occ_loss + lambda_q * q_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            run_loss += total_loss.item()
            run_occ  += occ_loss.item()
            run_q    += q_loss.item()
            nb += 1

            train_iter.set_postfix({
                "total": f"{total_loss.item():.3f}",
                "occ": f"{occ_loss.item():.3f}",
                "q": f"{q_loss.item():.3f}"
            })

        avg_loss = run_loss / max(nb,1)
        avg_occ  = run_occ  / max(nb,1)
        avg_q    = run_q    / max(nb,1)

        # 验证
        val_summary = evaluate_model(
            model, val_loader, device,
            quantiles=quantiles,
            tail_adjust_cfg=tail_adjust_cfg,
            occ_prob_threshold=args.occ_prob_threshold_eval,
            drizzle_cut=args.drizzle_cut
        )
        val_rmse = val_summary['Model']['RMSE']
        val_bias = val_summary['Model']['Bias']

        print(f"[Epoch {epoch}] TrainLoss={avg_loss:.4f} (occ={avg_occ:.3f}, q={avg_q:.3f})  "
              f"Val RMSE={val_rmse:.3f}  Val Bias={val_bias:.3f}")

        # 早停
        if val_rmse + 1e-4 < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            patience_counter = 0
            torch.save(best_state, os.path.join(output_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}  Best Val RMSE={best_val_rmse:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# =========================================================
# JSON Encoder
# =========================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

# =========================================================
# main
# =========================================================
def main():
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "train_config.json"), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, cls=NpEncoder)

    print("[INFO] 加载 ERA5...")
    era5_map = load_nc_date_var_map(ERA5_DIR, ERA5_VAR, time_name='date')
    print("[INFO] 加载 CRA...")
    cra_map = load_nc_date_var_map(CRA_DIR, CRA_VAR, time_name='time')
    print("[INFO] 加载 PSURF...")
    psurf_map = load_nc_date_var_map(PSURF_DIR, PSURF_VAR, time_name='time')
    print("[INFO] 加载 QAIR...")
    qair_map = load_nc_date_var_map(QAIR_DIR, QAIR_VAR, time_name='time')
    print("[INFO] 加载 DEM & SLOPE ...")
    dem_arr, dem_tr = load_dem_tif(DEM_PATH)
    slope_arr, slope_tr = load_dem_tif(SLOPE_PATH)
    print("[INFO] 加载站点 CSV...")
    station_df = pd.read_csv(STATION_CSV)
    station_df['date'] = pd.to_datetime(station_df['date'])
    print("[INFO] 加载 Niño 数据...")
    nino_map = load_nino_csv(NINO_CSV)

    station_df_train = station_df[(station_df['date'] >= TRAIN_START) &
                                  (station_df['date'] <= TRAIN_END)].copy()

    dataset = PrecipDatasetNC(
        era5_map, cra_map, psurf_map, qair_map,
        dem_arr, dem_tr, slope_arr,
        station_df_train,
        patch_size=args.patch_size,
        nino_map=nino_map,
        pad_mode='reflect'
    )
    dataset.export_samples_csv(os.path.join(args.output_dir, "train_samples_meta.csv"))

    n_total = len(dataset)
    idx_all = np.arange(n_total)
    np.random.shuffle(idx_all)
    val_size = int(n_total * args.val_ratio)
    val_idx = idx_all[:val_size]
    train_idx = idx_all[val_size:]

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    if args.use_weighted_sampler:
        print("[INFO] 使用 WeightedRandomSampler ...")
        obs_grades = dataset.obs_grade[train_idx]
        unique, counts = np.unique(obs_grades, return_counts=True)
        freq_map = {g: c for g, c in zip(unique, counts)}
        weights = np.array([1.0 / freq_map[g] for g in obs_grades], dtype=np.float32)
        weights = weights / weights.sum()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    quantiles = (0.5, 0.9, 0.95)
    model = MultiTaskQuantileCNN(in_channels=21, patch_size=args.patch_size, quantiles=quantiles)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] 使用设备:", device)
    model.to(device)

    tail_adjust_cfg = None
    if args.enable_tail_adjust:
        print(f"[INFO] 计算 tail ratio (>= {args.tail_threshold} mm)...")
        era5_train_vals = dataset.era5_center_vals[train_idx]
        obs_train_vals  = dataset.targets[train_idx]
        tail_ratio = compute_tail_ratio(era5_train_vals, obs_train_vals, threshold=args.tail_threshold)
        tail_adjust_cfg = {
            "ratio": tail_ratio,
            "threshold": args.tail_threshold,
            "width": args.tail_smooth_width
        }
        print(f"[INFO] Tail ratio = {tail_ratio:.3f}")

    model = train(model, train_loader, val_loader, device,
                  epochs=args.epochs, lr=args.lr,
                  quantiles=quantiles,
                  tail_adjust_cfg=None,   # 训练过程验证：先看无尾部性能
                  output_dir=args.output_dir,
                  lambda_occ=args.lambda_occ,
                  lambda_q=args.lambda_q)

    print("[INFO] 最终评估（无尾部调整）...")
    final_no_tail = evaluate_model(
        model, val_loader, device,
        quantiles=quantiles,
        tail_adjust_cfg=None,
        occ_prob_threshold=args.occ_prob_threshold_eval,
        drizzle_cut=args.drizzle_cut
    )
    with open(os.path.join(args.output_dir, "final_eval_no_tail.json"), 'w', encoding='utf-8') as f:
        json.dump(final_no_tail, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    if tail_adjust_cfg is not None:
        print("[INFO] 最终评估（含尾部调整）...")
        final_with_tail = evaluate_model(
            model, val_loader, device,
            quantiles=quantiles,
            tail_adjust_cfg=tail_adjust_cfg,
            occ_prob_threshold=args.occ_prob_threshold_eval,
            drizzle_cut=args.drizzle_cut
        )
        with open(os.path.join(args.output_dir, "final_eval_with_tail.json"), 'w', encoding='utf-8') as f:
            json.dump(final_with_tail, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    print("[DONE] 训练与评估完成。输出目录:", args.output_dir)

if __name__ == '__main__':
    main()