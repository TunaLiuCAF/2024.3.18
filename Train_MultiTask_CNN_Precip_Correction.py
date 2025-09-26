#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多任务降水校正训练脚本（基于你原始 simple CNN 的增强版）

核心改动:
1. 数据处理: 保留原逻辑, 但改进 patch 边缘处理, 增加观测分级.
2. 模型结构: 多任务 (发生概率 + 分位数残差).
3. 目标空间: 使用 log1p(precip) 减少重尾影响.
4. 损失函数: BCE(发生) + Pinball(分位数 q50,q90,q95).
5. 权重: 基于观测强度分级 (而不是 ERA5 等级, 避免 ERA5 错误放大).
6. 评估: 增加分箱 Bias/RMSE、阈值 (POD/FAR/CSI)、分位数统计.
7. 可选尾部平滑校正: 统计乘法比例 + 连续混合函数(避免 25mm 断点).
8. 高注释+结构化, 方便后续添加 CQM / 极值外推 / 时序特征.

运行方式:
python Train_MultiTask_CNN_Precip_Correction.py \
  --epochs 60 --batch_size 64 --patch_size 11 --lr 1e-3 \
  --output_dir ./output_new --enable_tail_adjust

注意:
- 路径常量沿用你的原始代码, 确认存在.
- 如果显存不足可调低 batch_size.
- 如果要加入更多环境特征（CAPE、IVT），可在 Dataset 中按通道追加。

Author: (改写增强) Copilot Assistant
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
parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例（随机拆分，后续可替换为时间或站点分割）')
parser.add_argument('--use_weighted_sampler', action='store_true', help='是否使用加权采样提升中高雨量出现频率')
parser.add_argument('--enable_tail_adjust', action='store_true', help='训练后是否启用尾部乘法平滑修正')
parser.add_argument('--tail_threshold', type=float, default=25.0, help='尾部修正起始阈值(mm)')
parser.add_argument('--tail_smooth_width', type=float, default=4.0, help='尾部平滑过渡宽度')
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
# 常量 / 归一化参数（保留原定义，可后续改为动态计算分位数）
# =========================================================
ERA5_MAX = 380.15     # 不直接用于回归（我们用 log1p），仍保留作特征归一化
CRA_MAX = 121.902878
PSURF_MIN, PSURF_MAX = 61681.609609, 101344.341562
QAIR_MIN, QAIR_MAX = 0.0, 0.024567
DEM_MIN, DEM_MAX = 429, 4439

# =========================================================
# 数据路径（与你原始脚本保持一致）
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
SLOPE_PATH = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/Terrain/SlopeD.tif"  # 注意：原文件似乎不是aspect
STATION_CSV = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/StationData_with_coords.csv"
NINO_CSV = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/nina34.anom.csv"

# 训练时间段（保持与你原文件一致，便于对比）
TRAIN_START = pd.to_datetime('2006-01-01')
TRAIN_END = pd.to_datetime('2010-12-31')

# =========================================================
# 基础工具函数
# =========================================================
def load_nc_date_var_map(nc_dir, varname, time_name=None):
    """
    读取目录下所有 nc 文件，按日期得到 2D 数组。
    与原脚本相同，仅添加少量健壮性注释。
    """
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
            # 回退：从文件名尝试解析日期
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
        transform = src.transform
        return arr, transform

def find_latlon_idx(transform, lat, lon):
    col, row = ~transform * (lon, lat)
    return int(round(row)), int(round(col))

def load_nino_csv(nino_csv_path):
    nino_df = pd.read_csv(nino_csv_path)
    nino_map = {}
    colnames = [c.lower() for c in nino_df.columns]
    if "year" in colnames and "month" in colnames:
        ycol = nino_df.columns[colnames.index("year")]
        mcol = nino_df.columns[colnames.index("month")]
        vcol = [c for c in nino_df.columns if c.lower() in ("value", "nino", "anom")][0]
        for _, row in nino_df.iterrows():
            period = pd.Period(year=int(row[ycol]), month=int(row[mcol]), freq='M')
            nino_map[period] = float(row[vcol])
    elif "date" in colnames:
        dcol = nino_df.columns[colnames.index("date")]
        vcol = [c for c in nino_df.columns if c.lower() in ("value", "nino", "anom")][0]
        for _, row in nino_df.iterrows():
            dt = pd.to_datetime(str(row[dcol]), errors='coerce')
            if pd.isna(dt):
                continue
            period = pd.Period(year=dt.year, month=dt.month, freq='M')
            nino_map[period] = float(row[vcol])
    else:
        raise ValueError("NINO CSV需有year/month/value或date/value列")
    return nino_map

# =========================================================
# 强度分级（观测与 ERA5 均可用，但权重基于观测更合理）
# =========================================================
def grade_precip(value):
    """
    可根据业务微调阈值；此处保持原始等级（与你原代码保持一致）：
    <0.1, 0.1-10, 10-25, 25-50, 50-100, 100-250, >=250
    返回 0...6
    """
    if value < 0.1: return 0
    elif value < 10: return 1
    elif value < 25: return 2
    elif value < 50: return 3
    elif value < 100: return 4
    elif value < 250: return 5
    else: return 6

def get_month_onehot(date):
    month = pd.to_datetime(date).month
    onehot = np.zeros(12, dtype=np.float32)
    onehot[month - 1] = 1
    return onehot

# =========================================================
# 数据集
# =========================================================
class PrecipDatasetNC(Dataset):
    """
    与原始版本对比的变化:
    1. patch 提取采用 reflect/edge 扩展，避免零填充边缘伪信息。
    2. 新增 obs_grade（用于加权），ERA5 等级仍可保留作为输入特征 (era5_grade_patch)。
    3. 仍采用 "中心像素 ERA5" 作为基准输入（residual学 log 空间差）。
    4. NiÑo, month one-hot 保留。
    """
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

        print("[INFO] 准备采样 patch ...")
        for _, row in tqdm(station_df.iterrows(), total=len(station_df)):
            date_str = pd.to_datetime(row['date']).strftime("%Y-%m-%d")
            lat, lon = row['y'], row['x']
            if (date_str not in era5_map or date_str not in cra_map or
                date_str not in psurf_map or date_str not in qair_map or
                np.isnan(row['Prcp'])):
                continue

            # 找到地形 pixel 索引
            row_idx, col_idx = find_latlon_idx(self.dem_tr, lat, lon)

            # 提取 patch
            def extract_patch(arr):
                """使用反射或边缘 padding 获取固定尺寸 patch"""
                half = self.patch_size // 2
                # pad
                if self.pad_mode == 'reflect':
                    padded = np.pad(arr, pad_width=half, mode='reflect')
                else:
                    padded = np.pad(arr, pad_width=half, mode='edge')
                r = row_idx + half
                c = col_idx + half
                patch = padded[r - half:r + half + 1, c - half:c + half + 1]
                if patch.shape != (self.patch_size, self.patch_size):
                    # 冗余保护
                    return None
                return patch

            era5_patch = extract_patch(self.era5_map[date_str])
            cra_patch = extract_patch(self.cra_map[date_str])
            psurf_patch = extract_patch(self.psurf_map[date_str])
            qair_patch = extract_patch(self.qair_map[date_str])
            dem_patch = extract_patch(self.dem_arr)
            slope_patch = extract_patch(self.slope_arr)
            if any(p is None for p in [era5_patch, cra_patch, psurf_patch, qair_patch, dem_patch, slope_patch]):
                continue

            # slope 文件原用途不明（你原代码当作 "aspect" 做 sin/cos）
            # 这里保留你的处理方式，只改名为 slope_aspect 以提醒。
            aspect_rad = np.deg2rad(slope_patch)
            aspect_sin = np.sin(aspect_rad)
            aspect_cos = np.cos(aspect_rad)

            # ERA5 等级图（作为特征）
            era5_grade_patch = np.vectorize(grade_precip)(era5_patch).astype(np.float32) / 6.0

            # 归一化
            era5_patch_norm = np.clip(era5_patch / ERA5_MAX, 0, 1)
            cra_patch_norm = np.clip(cra_patch / CRA_MAX, 0, 1)
            psurf_patch_norm = np.clip((psurf_patch - PSURF_MIN) / (PSURF_MAX - PSURF_MIN), 0, 1)
            qair_patch_norm = np.clip((qair_patch - QAIR_MIN) / (QAIR_MAX - QAIR_MIN), 0, 1)
            dem_patch_norm = np.clip((dem_patch - DEM_MIN) / (DEM_MAX - DEM_MIN), 0, 1)

            # 月份 one-hot
            month_onehot = get_month_onehot(date_str)
            month_channels = month_onehot[:, None, None] * np.ones((12, self.patch_size, self.patch_size), dtype=np.float32)

            # NiÑo 指标
            def get_nino_for_date(date):
                period = pd.to_datetime(date).to_period('M')
                return self.nino_map.get(period, 0.0) if self.nino_map is not None else 0.0
            nino_value = get_nino_for_date(date_str)
            nino_layer = np.ones((self.patch_size, self.patch_size), dtype=np.float32) * nino_value

            # 组织特征通道 (与原先 21 通道结构保持兼容 + 你可在此追加更多物理量)
            patch = np.concatenate([
                np.stack([
                    era5_patch_norm, cra_patch_norm, dem_patch_norm,
                    aspect_sin, aspect_cos,
                    psurf_patch_norm, qair_patch_norm, era5_grade_patch
                ], axis=0),
                month_channels,
                nino_layer[None, ...]
            ], axis=0)  # shape (21, H, W)

            if np.any(np.isnan(patch)):
                continue

            obs_precip = float(row['Prcp'])
            era5_center = float(era5_patch[self.patch_size // 2, self.patch_size // 2])
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
        self.targets = np.array(self.targets, dtype=np.float32)
        self.era5_center_vals = np.array(self.era5_center_vals, dtype=np.float32)
        self.era5_center_grade = np.array(self.era5_center_grade, dtype=np.int64)
        self.obs_grade = np.array(self.obs_grade, dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 返回：特征, 观测降水, ERA5 原值, ERA5 等级(可选), 观测等级(用于权重)
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.era5_center_vals[idx], dtype=torch.float32),
            torch.tensor(self.era5_center_grade[idx], dtype=torch.long),
            torch.tensor(self.obs_grade[idx], dtype=torch.long)
        )

    def export_samples_csv(self, out_csv):
        df = pd.DataFrame(self.meta_rows)
        df.to_csv(out_csv, index=False)
        print(f"[INFO] 导出样本元数据 -> {out_csv}, 共 {len(df)} 条")

# =========================================================
# 多任务模型 (Occurrence + 分位数残差)
# =========================================================
class MultiTaskQuantileCNN(nn.Module):
    """
    输出:
      occ_logit: 是否 >0.1 mm 的 logits
      resid_q:   对应分位数的 log 残差 (与 era5_log 相加 -> 预测 log 分位)
    """
    def __init__(self, in_channels, patch_size, quantiles=(0.5, 0.9, 0.95)):
        super().__init__()
        self.quantiles = quantiles

        # 基础卷积骨干 (可以后续替换为 UNet / Dilated / Attention)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),  # 若 batch 很小可考虑改 GroupNorm
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (patch/2)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (patch/4)

            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> (patch/8)
        )

        # 自适应池化避免 patch_size 非 8 的倍数问题
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat_dim = 96

        # 发生概率头
        self.occ_head = nn.Linear(feat_dim, 1)
        # 分位数残差头 (输出每个 quantile 的残差, log 空间)
        self.resid_head = nn.Linear(feat_dim, len(self.quantiles))

    def forward(self, x):
        h = self.backbone(x)
        h = self.gap(h).flatten(1)  # [B, feat_dim]
        occ_logit = self.occ_head(h).squeeze(1)   # [B]
        resid_q = self.resid_head(h)              # [B, n_quantiles]
        return occ_logit, resid_q

# =========================================================
# 分位数 (Pinball) 损失
# =========================================================
def pinball_loss(pred, target, q):
    """
    pred/target: shape [N]
    q: 分位值 (0<q<1)
    """
    diff = target - pred
    return torch.mean(torch.maximum(q * diff, (q - 1) * diff))

def multi_quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        losses.append(pinball_loss(preds[:, i], target, q))
    return sum(losses) / len(losses)

# =========================================================
# 强度权重（观测分级） - 可根据需要微调
# =========================================================
def build_intensity_weights(obs_grade_tensor):
    """
    给不同观测强度等级设置权重：
    等级: 0,1,2,3,4,5,6
    想法: 增强中高雨的学习信号，但避免让模型只盯极端（梯度爆）
    可以根据训练集样本占比再自适应调整。
    """
    base_weights = torch.tensor([1.0, 1.2, 1.4, 1.8, 2.2, 2.8, 3.2], device=obs_grade_tensor.device)
    return base_weights[obs_grade_tensor]

# =========================================================
# 阈值事件评分
# =========================================================
def event_scores(pred, obs, thresholds=(0.1, 1, 10, 25, 50)):
    """
    返回 {thr: {POD, FAR, CSI, Hits, Miss, FalseAlarms}}.
    """
    results = {}
    for thr in thresholds:
        pred_event = pred >= thr
        obs_event = obs >= thr
        hits = np.sum(pred_event & obs_event)
        miss = np.sum(~pred_event & obs_event)
        false = np.sum(pred_event & ~obs_event)
        # POD
        pod = hits / (hits + miss + 1e-9)
        # FAR
        far = false / (hits + false + 1e-9)
        # CSI
        csi = hits / (hits + miss + false + 1e-9)
        results[thr] = {
            "POD": float(pod),
            "FAR": float(far),
            "CSI": float(csi),
            "Hits": int(hits),
            "Miss": int(miss),
            "FalseAlarms": int(false)
        }
    return results

# =========================================================
# 分箱 Bias / RMSE
# =========================================================
def bin_stats(pred, obs, bins=(0,1,5,10,20,30,50,100,1e9)):
    idx = np.digitize(obs, bins) - 1  # 0..len(bins)-2
    records = []
    for i in range(len(bins)-1):
        mask = idx == i
        if np.sum(mask) == 0:
            continue
        p = pred[mask]
        o = obs[mask]
        bias = np.mean(p - o)
        rmse = math.sqrt(mean_squared_error(o, p))
        records.append({
            "bin": f"{bins[i]}-{bins[i+1]}",
            "n": int(np.sum(mask)),
            "bias": float(bias),
            "rmse": float(rmse),
            "obs_mean": float(np.mean(o)),
            "pred_mean": float(np.mean(p))
        })
    return records

# =========================================================
# 尾部平滑修正函数（可选）
# =========================================================
def compute_tail_ratio(era5_vals, obs_vals, threshold=25.0):
    """
    计算训练集中 ERA5 vs 观测 在 >= threshold 区域的中位数比例 (Obs / ERA5)。
    防止极端点 ERA5 很小 -> 比例爆炸, 先筛掉 ERA5<5 的异常。
    """
    mask = (obs_vals >= threshold) & (era5_vals >= 5)
    if np.sum(mask) < 5:
        print("[WARN] 高强度样本太少，tail_ratio 回退为 1.0")
        return 1.0
    ratios = obs_vals[mask] / np.maximum(era5_vals[mask], 1e-6)
    # 中位数更稳健
    return float(np.median(ratios))

def smooth_weight(values, threshold, width):
    """
    平滑混合权重 w(I):
    w = 0.5 * (1 + tanh((I - threshold)/width))
    I >> threshold -> w -> 1
    I << threshold -> w -> 0
    """
    return 0.5 * (1 + np.tanh((values - threshold) / (width + 1e-6)))

def apply_tail_adjustment(raw_pred, era5_raw, tail_ratio, threshold, width):
    """
    raw_pred: 模型预测（校正后）
    era5_raw: 原 ERA5
    tail_ratio: >=threshold 估计得到的缩放比例
    threshold, width: 平滑过渡参数
    返回：平滑混合 corrected
    extreme_adjust = era5_raw * tail_ratio
    corrected = (1 - w)*raw_pred + w*extreme_adjust
    """
    w = smooth_weight(era5_raw, threshold, width)
    extreme_adjust = era5_raw * tail_ratio
    corrected = (1 - w) * raw_pred + w * extreme_adjust
    return corrected

# =========================================================
# 评估函数
# =========================================================
def evaluate_model(model, dataloader, device, quantiles=(0.5,0.9,0.95),
                   tail_adjust_cfg=None):
    model.eval()
    all_obs = []
    all_pred_point = []
    all_era5 = []

    with torch.no_grad():
        for X, y, era5_c, era5_grade, obs_grade in dataloader:
            X = X.to(device)
            y_np = y.numpy()
            era5_np = era5_c.numpy()
            # 前向
            occ_logit, resid_q = model(X)
            # resid_q shape [B, n_q] -> log 残差
            era5_log = torch.log1p(era5_c.to(device).clamp(min=0))
            pred_log_q = era5_log.unsqueeze(1) + resid_q  # [B, n_q]
            # 取中位数 q50 作为点估计
            # 假设 quantiles[0] == 0.5
            mid_idx = quantiles.index(0.5)
            pred_mid_log = pred_log_q[:, mid_idx]
            pred_mid = torch.expm1(pred_mid_log).cpu().numpy()

            # 可选尾部调整
            if tail_adjust_cfg is not None:
                pred_mid = apply_tail_adjustment(
                    pred_mid,
                    era5_np,
                    tail_adjust_cfg['ratio'],
                    tail_adjust_cfg['threshold'],
                    tail_adjust_cfg['width']
                )

            all_obs.append(y_np)
            all_pred_point.append(pred_mid)
            all_era5.append(era5_np)

    obs = np.concatenate(all_obs)
    pred = np.concatenate(all_pred_point)
    era5_vals = np.concatenate(all_era5)

    # 过滤 NaN（理论上没有）
    m = ~np.isnan(obs) & ~np.isnan(pred)
    obs = obs[m]
    pred = pred[m]
    era5_vals = era5_vals[m]

    # 指标
    bias = float(np.mean(pred - obs))
    rmse = math.sqrt(mean_squared_error(obs, pred))
    corr = pearsonr(pred, obs)[0] if len(obs) > 2 else float('nan')

    # ERA5 baseline
    era5_bias = float(np.mean(era5_vals - obs))
    era5_rmse = math.sqrt(mean_squared_error(obs, era5_vals))
    era5_corr = pearsonr(era5_vals, obs)[0] if len(obs) > 2 else float('nan')

    # 分箱
    bins_result = bin_stats(pred, obs)

    # 阈值事件
    evt_scores = event_scores(pred, obs)

    # 分位数对比（用于查看分布形态）
    quantile_points = [0.5, 0.75, 0.9, 0.95, 0.99]
    q_table = []
    for q in quantile_points:
        q_table.append({
            "q": q,
            "Obs": float(np.quantile(obs, q)),
            "Pred": float(np.quantile(pred, q)),
            "ERA5": float(np.quantile(era5_vals, q))
        })

    summary = {
        "Model": {"Bias": bias, "RMSE": rmse, "Corr": corr},
        "ERA5": {"Bias": era5_bias, "RMSE": era5_rmse, "Corr": era5_corr},
        "Bins": bins_result,
        "Events": evt_scores,
        "Quantiles": q_table,
        "N": int(len(obs))
    }
    return summary

# =========================================================
# 训练主循环
# =========================================================
def train(model, train_loader, val_loader, device, epochs, lr,
          quantiles=(0.5,0.9,0.95),
          tail_adjust_cfg=None,
          output_dir='./output'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_rmse = float('inf')
    best_state = None
    patience = 8
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_occ = 0.0
        running_q = 0.0
        count_batches = 0

        train_iter = tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False)
        for (X, y, era5_c, era5_grade, obs_grade) in train_iter:
            X = X.to(device)
            y = y.to(device)
            era5_c = era5_c.to(device)
            obs_grade = obs_grade.to(device)

            # -----------------------------
            # 构造标签
            # -----------------------------
            occ_label = (y > 0.1).float()

            # log1p
            y_log = torch.log1p(y)
            era5_log = torch.log1p(era5_c.clamp(min=0))

            # -----------------------------
            # 前向
            # -----------------------------
            occ_logit, resid_q = model(X)
            # resid_q: [B, n_q] -> log 残差
            pred_log_q = era5_log.unsqueeze(1) + resid_q  # 预测 log 分位

            # -----------------------------
            # 损失：Occurrence + Quantile
            # -----------------------------
            occ_loss = F.binary_cross_entropy_with_logits(occ_logit, occ_label)

            # 仅对雨日样本计算分位数损失（可选：对所有样本也算，只是 y=0 会收缩）
            rain_mask = occ_label > 0
            if rain_mask.sum() > 0:
                # 取 predicted log 分位 -> 反变换或直接与 y_log 比较?
                # 这里直接用 log 空间 pinball, target=y_log
                pred_log_q_rain = pred_log_q[rain_mask]
                y_log_rain = y_log[rain_mask]
                q_loss = multi_quantile_loss(pred_log_q_rain, y_log_rain, quantiles)
            else:
                q_loss = torch.tensor(0.0, device=device)

            # 强度加权（只对 q_loss 部分加权更自然，也可整合）
            weights = build_intensity_weights(obs_grade)
            # 让权重在无降水样本上=1（避免无雨权重漂移）
            weights = torch.where(rain_mask, weights, torch.ones_like(weights))
            # 平均权重
            w_mean = torch.mean(weights)
            # 这里简单乘到分位数损失上
            total_loss = occ_loss + (q_loss * w_mean)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_occ += occ_loss.item()
            running_q += q_loss.item()
            count_batches += 1
            train_iter.set_postfix({
                "loss": f"{total_loss.item():.3f}",
                "occ": f"{occ_loss.item():.3f}",
                "q": f"{q_loss.item():.3f}"
            })

        avg_loss = running_loss / max(count_batches, 1)
        avg_occ = running_occ / max(count_batches, 1)
        avg_q = running_q / max(count_batches, 1)

        # -----------------------------
        # 验证阶段
        # -----------------------------
        val_summary = evaluate_model(
            model, val_loader, device,
            quantiles=quantiles,
            tail_adjust_cfg=tail_adjust_cfg  # 验证时也可查看加尾部后的指标
        )
        val_rmse = val_summary['Model']['RMSE']
        print(f"[Epoch {epoch}] TrainLoss={avg_loss:.4f} (occ={avg_occ:.3f}, q={avg_q:.3f})  "
              f"Val RMSE={val_rmse:.3f}  Val Bias={val_summary['Model']['Bias']:.3f}")

        # 早停逻辑
        if val_rmse + 1e-4 < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            patience_counter = 0
            torch.save(best_state, os.path.join(output_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}. Best Val RMSE={best_val_rmse:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# =========================================================
# JSON 序列化兼容 numpy
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
# 主执行
# =========================================================
def main():
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置 (加 cls=NpEncoder)
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
    print("[INFO] 加载 DEM...")
    dem_arr, dem_tr = load_dem_tif(DEM_PATH)
    print("[INFO] 加载 SLOPE (作为 Aspect 使用)...")
    slope_arr, slope_tr = load_dem_tif(SLOPE_PATH)
    print("[INFO] 加载站点 CSV...")
    station_df = pd.read_csv(STATION_CSV)
    station_df['date'] = pd.to_datetime(station_df['date'])
    print("[INFO] 加载 Niño 数据...")
    nino_map = load_nino_csv(NINO_CSV)

    # 过滤训练时间段
    station_df_train = station_df[
        (station_df['date'] >= TRAIN_START) & (station_df['date'] <= TRAIN_END)
    ].copy()

    # 构建 Dataset
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
    val_set = Subset(dataset, val_idx)

    # 采样器（可选）：提升高强度样本出现频率
    if args.use_weighted_sampler:
        print("[INFO] 使用 WeightedRandomSampler 增强中高雨量样本频率...")
        # 基于 obs_grade 分配反频率权重
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

    # 模型
    quantiles = (0.5, 0.9, 0.95)
    model = MultiTaskQuantileCNN(in_channels=21, patch_size=args.patch_size, quantiles=quantiles)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] 使用设备:", device)
    model = model.to(device)

    # (可选) 先统计 tail ratio，用于训练过程中的验证集评估或训练后最终输出
    tail_adjust_cfg = None
    if args.enable_tail_adjust:
        print("[INFO] 预计算尾部比例 (>= {:.1f} mm) ...".format(args.tail_threshold))
        # 取训练集中 ERA5 + 观测
        era5_train_vals = dataset.era5_center_vals[train_idx]
        obs_train_vals = dataset.targets[train_idx]
        tail_ratio = compute_tail_ratio(era5_train_vals, obs_train_vals, threshold=args.tail_threshold)
        tail_adjust_cfg = {
            "ratio": tail_ratio,
            "threshold": args.tail_threshold,
            "width": args.tail_smooth_width
        }
        print(f"[INFO] Tail ratio = {tail_ratio:.3f}")

    # 训练
    model = train(model, train_loader, val_loader, device,
                  epochs=args.epochs, lr=args.lr,
                  quantiles=quantiles,
                  tail_adjust_cfg=tail_adjust_cfg,
                  output_dir=args.output_dir)

    # 最终评估（含 tail 调整 & 不含 tail 调整 都给一份）
    print("[INFO] 最终评估（无尾部调整）...")
    final_no_tail = evaluate_model(model, val_loader, device, quantiles, tail_adjust_cfg=None)
    with open(os.path.join(args.output_dir, "final_eval_no_tail.json"), 'w', encoding='utf-8') as f:
        json.dump(final_no_tail, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    if tail_adjust_cfg is not None:
        print("[INFO] 最终评估（含尾部调整）...")
        final_with_tail = evaluate_model(model, val_loader, device, quantiles, tail_adjust_cfg=tail_adjust_cfg)
        with open(os.path.join(args.output_dir, "final_eval_with_tail.json"), 'w', encoding='utf-8') as f:
            json.dump(final_with_tail, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    print("[DONE] 训练与评估完成。输出目录:", args.output_dir)

if __name__ == '__main__':
    main()