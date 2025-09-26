#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多任务降水校正 推理脚本 (与最新训练脚本匹配版本)

匹配点：
1. 训练时模型输出: occ_logit（是否下雨概率的 logit） + resid_q（各分位 log 残差）。
2. 训练时：log1p(ERA5中心) + resid -> 预测各分位的 log 值；推理保持同样组合后 expm1 还原。
3. 推理后处理顺序与训练思想一致：
   (1) 概率阈值裁剪  p < occ_prob_threshold -> 设 0
   (2) 毛毛雨截断    0 < 预测 < drizzle_cut  -> 设 0
   (3) (可选) 尾部平滑放大  (在过滤后执行，避免放大小噪声)
4. 中位数分位 quantile_mid 通常为 0.5 (q50) 对应训练中的第一个分位。
5. 对 q90 / q95 同步做概率裁剪与毛毛雨截断，保持分位数关系合理。
6. NetCDF 属性写入前做类型清洗（避免 numpy.bool_ 等导致写文件报错）。
7. 属性中记录 occ_prob_threshold 与 drizzle_cut，方便复现。

使用示例：
python Correct_era5_mj_cnn_multitask.py \
  --model_path ./output_multitask/best_model.pth \
  --year_start 2011 --year_end 2012 \
  --output_dir ./corrected_2011_2012 \
  --batch_size 512 \
  --occ_prob_threshold 0.5 \
  --drizzle_cut 0.15 \
  --enable_tail_adjust \
  --tail_ratio 1.30 \
  --tail_threshold 25 --tail_smooth_width 4

建议流程：
  先不加 --enable_tail_adjust 验证假湿是否下降；
  再加尾部调整提升中大雨。
"""

import os
import json
import argparse
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from tqdm import tqdm

import torch
import torch.nn as nn


# ==============================
# 命令行参数
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True,
                    help='训练得到的多任务模型路径 (best_model.pth)')
parser.add_argument('--output_dir', type=str, default='./corrected_output', help='输出目录')
parser.add_argument('--year_start', type=int, required=True, help='开始年份')
parser.add_argument('--year_end', type=int, required=True, help='结束年份 (包含)')
parser.add_argument('--patch_size', type=int, default=11, help='Patch 尺寸 (需与训练一致)')
parser.add_argument('--batch_size', type=int, default=256, help='推理批大小 (像元数)')
parser.add_argument('--device', type=str, default='auto', help='cuda / cpu / auto')

parser.add_argument('--occ_prob_threshold', type=float, default=0.5,
                    help='发生概率低于该阈值的像元置 0 (与训练评估 occ_prob_threshold_eval 对应)')
parser.add_argument('--drizzle_cut', type=float, default=0.15,
                    help='0 < 预测 < drizzle_cut 视为 0 的截断 (去毛毛雨)')

parser.add_argument('--quantile_mid', type=float, default=0.5,
                    help='点估计使用的分位 (需在模型 quantiles 中)')
parser.add_argument('--enable_tail_adjust', action='store_true',
                    help='是否启用尾部平滑乘法修正')
parser.add_argument('--tail_ratio', type=float, default=1.0,
                    help='≥ tail_threshold 区域混合向 ERA5*ratio 过渡 (来自训练统计 Obs/ERA5 中位数)')
parser.add_argument('--tail_threshold', type=float, default=25.0,
                    help='尾部平滑起始阈值 (mm)')
parser.add_argument('--tail_smooth_width', type=float, default=4.0,
                    help='平滑过渡宽度 (越大越平缓)')

parser.add_argument('--save_intermediate_json', action='store_true',
                    help='是否保存每年简单统计 JSON (分位数)')

args = parser.parse_args()


# ==============================
# 数据路径常量
# ==============================
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
NINO_CSV = "/Volumes/T7/master/CNN_Precipitation_Correction/03-NewBJ-MJ/UsefulData/nina34.anom.csv"

# ==============================
# 归一化参数 (与训练一致)
# ==============================
ERA5_MAX = 380.15
CRA_MAX = 121.902878
PSURF_MIN, PSURF_MAX = 61681.609609, 101344.341562
QAIR_MIN, QAIR_MAX = 0.0, 0.024567
DEM_MIN, DEM_MAX = 429, 4439


# ==============================
# 模型定义 (与训练结构一致)
# ==============================
class MultiTaskQuantileCNN(nn.Module):
    def __init__(self, in_channels: int, patch_size: int,
                 quantiles=(0.5, 0.9, 0.95)):
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
        self.occ_head = nn.Linear(feat_dim, 1)
        self.resid_head = nn.Linear(feat_dim, len(self.quantiles))

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        h = self.gap(h).flatten(1)
        occ_logit = self.occ_head(h).squeeze(1)
        resid_q = self.resid_head(h)
        return occ_logit, resid_q


# ==============================
# 工具函数
# ==============================
def load_tif(path: str):
    with rasterio.open(path) as src:
        return src.read(1)

def get_nc_var(ds: xr.Dataset, candidates):
    for v in candidates:
        if v in ds.variables:
            return ds[v]
    return list(ds.data_vars.values())[0]

def get_month_onehot(date_str: str) -> np.ndarray:
    m = pd.to_datetime(date_str).month
    arr = np.zeros(12, dtype=np.float32)
    arr[m-1] = 1
    return arr

def grade_precip(v: float) -> int:
    if v < 0.1: return 0
    elif v < 10: return 1
    elif v < 25: return 2
    elif v < 50: return 3
    elif v < 100: return 4
    elif v < 250: return 5
    else: return 6

def build_padding(arr: np.ndarray, pad: int, mode='reflect'):
    return np.pad(arr, pad_width=pad, mode=mode)

def load_nino_map(path: str) -> Dict[pd.Period, float]:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = value_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ('date', 'time'): date_col = c
        if lc in ('nino', 'value', 'anom'): value_col = c
    if date_col is None or value_col is None:
        raise ValueError("NINO 文件需含 date 与 nino/value/anom 列")
    out = {}
    for _, r in df.iterrows():
        dt = pd.to_datetime(str(r[date_col]), errors='coerce')
        if pd.isna(dt): continue
        out[dt.to_period('M')] = float(r[value_col])
    return out

def prepare_padded_fields(era5_day, cra_day, psurf_day, qair_day, dem, slope, pad, mode='reflect'):
    return (
        build_padding(era5_day, pad, mode),
        build_padding(cra_day, pad, mode),
        build_padding(psurf_day, pad, mode),
        build_padding(qair_day, pad, mode),
        build_padding(dem, pad, mode),
        build_padding(slope, pad, mode)
    )

def extract_all_patches(era5_p, cra_p, psurf_p, qair_p, dem_p, slope_p, patch_size):
    H = era5_p.shape[0] - (patch_size - 1)
    W = era5_p.shape[1] - (patch_size - 1)
    N = H * W
    ps = patch_size

    def collect(arr_p):
        out = np.zeros((N, ps, ps), dtype=arr_p.dtype)
        k = 0
        for i in range(H):
            for j in range(W):
                out[k] = arr_p[i:i+ps, j:j+ps]
                k += 1
        return out

    return {
        "era5": collect(era5_p),
        "cra": collect(cra_p),
        "psurf": collect(psurf_p),
        "qair": collect(qair_p),
        "dem": collect(dem_p),
        "slope": collect(slope_p)
    }

def build_feature_tensor(field_patches: Dict[str, np.ndarray],
                         month_onehot: np.ndarray,
                         nino_value: float) -> np.ndarray:
    era5_patch = field_patches['era5']
    cra_patch  = field_patches['cra']
    dem_patch  = field_patches['dem']
    slope_patch = field_patches['slope']
    psurf_patch = field_patches['psurf']
    qair_patch  = field_patches['qair']

    aspect_rad = np.deg2rad(slope_patch)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    era5_grade_patch = np.vectorize(grade_precip)(era5_patch).astype(np.float32) / 6.0

    era5_norm  = np.clip(era5_patch / ERA5_MAX, 0, 1)
    cra_norm   = np.clip(cra_patch / CRA_MAX, 0, 1)
    dem_norm   = np.clip((dem_patch - DEM_MIN)/(DEM_MAX-DEM_MIN), 0, 1)
    psurf_norm = np.clip((psurf_patch - PSURF_MIN)/(PSURF_MAX-PSURF_MIN), 0, 1)
    qair_norm  = np.clip((qair_patch - QAIR_MIN)/(QAIR_MAX-QAIR_MIN), 0, 1)

    N, ps, _ = era5_norm.shape
    month_channels = month_onehot[:, None, None] * np.ones((12, ps, ps), dtype=np.float32)
    month_channels = np.repeat(month_channels[None, ...], N, axis=0)
    nino_layer = np.ones((N, 1, ps, ps), dtype=np.float32) * nino_value

    base_stack = np.stack([
        era5_norm, cra_norm, dem_norm,
        aspect_sin, aspect_cos,
        psurf_norm, qair_norm, era5_grade_patch
    ], axis=1)

    return np.concatenate([base_stack, month_channels, nino_layer], axis=1)

def apply_tail_adjustment(pred_mid: np.ndarray,
                          era5_center: np.ndarray,
                          tail_ratio: float,
                          threshold: float,
                          width: float) -> np.ndarray:
    # 平滑权重
    w = 0.5 * (1 + np.tanh((era5_center - threshold) / (width + 1e-6)))
    extreme = era5_center * tail_ratio
    return (1 - w) * pred_mid + w * extreme

def sanitize_netcdf_attrs(ds: xr.Dataset) -> xr.Dataset:
    import numpy as np
    import json as _json
    new_attrs = {}
    for k, v in ds.attrs.items():
        if isinstance(v, (np.bool_, bool)):
            new_attrs[k] = int(v)
        elif isinstance(v, (np.integer,)):
            new_attrs[k] = int(v)
        elif isinstance(v, (np.floating,)):
            new_attrs[k] = float(v)
        elif isinstance(v, (list, tuple)):
            new_attrs[k] = ",".join(map(str, v))
        elif isinstance(v, dict):
            new_attrs[k] = _json.dumps(v, ensure_ascii=False)
        else:
            new_attrs[k] = str(v)
    ds.attrs = new_attrs
    return ds


# ==============================
# 单年份校正
# ==============================
def correct_year(year: int,
                 model: MultiTaskQuantileCNN,
                 dem_arr: np.ndarray,
                 slope_arr: np.ndarray,
                 nino_map: Dict[pd.Period, float],
                 device: torch.device,
                 quantiles=(0.5, 0.9, 0.95),
                 batch_size=256,
                 patch_size=11,
                 occ_prob_threshold=0.5,
                 drizzle_cut=0.15,
                 enable_tail=False,
                 tail_ratio=1.0,
                 tail_threshold=25.0,
                 tail_smooth_width=4.0,
                 quantile_mid=0.5) -> xr.Dataset:

    # --- 路径检查 ---
    era5_path = os.path.join(ERA5_DIR, f'ERA5_daily_precip_BJ_{year}_mm.nc')
    cra_path  = os.path.join(CRA_DIR, f'CRA_TotalPrecip_tavg_{year}_ERA5_extent.nc')
    psurf_path = os.path.join(PSURF_DIR, f'CRA_Psurf_f_inst_{year}_ERA5_extent.nc')
    qair_path  = os.path.join(QAIR_DIR, f'CRA_Qair_f_inst_{year}_ERA5_extent.nc')
    for p in [era5_path, cra_path, psurf_path, qair_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少输入文件: {p}")

    # --- 打开数据 ---
    era5_ds = xr.open_dataset(era5_path)
    era5_tp = get_nc_var(era5_ds, [ERA5_VAR, 'tp']).values  # (T,H,W)
    lat = era5_ds['lat'].values if 'lat' in era5_ds else era5_ds['latitude'].values
    lon = era5_ds['lon'].values if 'lon' in era5_ds else era5_ds['longitude'].values
    dates = era5_ds['time'].values if 'time' in era5_ds else era5_ds['date'].values

    cra_tp = get_nc_var(xr.open_dataset(cra_path), [CRA_VAR]).values
    psurf  = get_nc_var(xr.open_dataset(psurf_path), [PSURF_VAR]).values
    qair   = get_nc_var(xr.open_dataset(qair_path), [QAIR_VAR]).values

    T, H, W = era5_tp.shape
    ps = patch_size
    pad = ps // 2

    # --- 输出容器 ---
    corrected_mid = np.zeros_like(era5_tp, dtype=np.float32)
    corrected_q90 = np.zeros_like(era5_tp, dtype=np.float32)
    corrected_q95 = np.zeros_like(era5_tp, dtype=np.float32)
    occ_prob_all  = np.zeros_like(era5_tp, dtype=np.float32)
    era5_raw      = era5_tp.astype(np.float32)

    if quantile_mid not in quantiles:
        raise ValueError(f"quantile_mid={quantile_mid} 不在 quantiles {quantiles} 中")

    q_index_mid = quantiles.index(quantile_mid)
    q_index_90  = quantiles.index(0.9) if 0.9 in quantiles else None
    q_index_95  = quantiles.index(0.95) if 0.95 in quantiles else None

    model.eval()
    with torch.no_grad():
        for t in tqdm(range(T), desc=f"Year {year} 推理"):
            date_str = pd.to_datetime(str(dates[t])).strftime("%Y-%m-%d")
            period = pd.to_datetime(date_str).to_period('M')
            nino_value = nino_map.get(period, 0.0)

            era5_day  = era5_tp[t]
            cra_day   = cra_tp[t]
            psurf_day = psurf[t]
            qair_day  = qair[t]

            era5_p, cra_p, psurf_p, qair_p, dem_p, slope_p = prepare_padded_fields(
                era5_day, cra_day, psurf_day, qair_day, dem_arr, slope_arr,
                pad=pad, mode='reflect'
            )
            patches_dict = extract_all_patches(era5_p, cra_p, psurf_p, qair_p, dem_p, slope_p, ps)
            month_onehot = get_month_onehot(date_str)
            X_np = build_feature_tensor(patches_dict, month_onehot, nino_value)
            era5_centers = patches_dict['era5'][:, ps//2, ps//2]  # (N,)

            preds_mid_list = []
            preds_q90_list = []
            preds_q95_list = []
            occ_list       = []

            N = X_np.shape[0]
            for start in range(0, N, batch_size):
                end = start + batch_size
                X_batch = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)
                era5_center_batch = torch.tensor(era5_centers[start:end], dtype=torch.float32, device=device)

                occ_logit, resid_q = model(X_batch)
                p = torch.sigmoid(occ_logit).cpu().numpy()

                era5_log = torch.log1p(era5_center_batch.clamp(min=0))
                pred_log_q = era5_log.unsqueeze(1) + resid_q         # [B,n_q]
                pred_q = torch.expm1(pred_log_q).cpu().numpy()       # 还原 mm

                # 中位数
                pred_mid = pred_q[:, q_index_mid]
                # 概率阈值
                if occ_prob_threshold > 0:
                    pred_mid = np.where(p >= occ_prob_threshold, pred_mid, 0.0)
                # 毛毛雨截断
                if drizzle_cut > 0:
                    pred_mid = np.where((pred_mid > 0) & (pred_mid < drizzle_cut), 0.0, pred_mid)

                preds_mid_list.append(pred_mid)
                occ_list.append(p)

                # 其他分位（保持与中位相同的裁剪逻辑）
                if q_index_90 is not None:
                    q90_vals = pred_q[:, q_index_90]
                    if occ_prob_threshold > 0:
                        q90_vals = np.where(p >= occ_prob_threshold, q90_vals, 0.0)
                    if drizzle_cut > 0:
                        q90_vals = np.where((q90_vals > 0) & (q90_vals < drizzle_cut), 0.0, q90_vals)
                    preds_q90_list.append(q90_vals)

                if q_index_95 is not None:
                    q95_vals = pred_q[:, q_index_95]
                    if occ_prob_threshold > 0:
                        q95_vals = np.where(p >= occ_prob_threshold, q95_vals, 0.0)
                    if drizzle_cut > 0:
                        q95_vals = np.where((q95_vals > 0) & (q95_vals < drizzle_cut), 0.0, q95_vals)
                    preds_q95_list.append(q95_vals)

            pred_mid_all = np.concatenate(preds_mid_list)
            occ_prob_all_day = np.concatenate(occ_list)
            pred_q90_all = (np.concatenate(preds_q90_list)
                            if q_index_90 is not None else np.full_like(pred_mid_all, np.nan))
            pred_q95_all = (np.concatenate(preds_q95_list)
                            if q_index_95 is not None else np.full_like(pred_mid_all, np.nan))

            # 尾部平滑（放在全部过滤后）
            if enable_tail and tail_ratio != 1.0:
                pred_mid_all = apply_tail_adjustment(pred_mid_all, era5_centers,
                                                     tail_ratio, tail_threshold, tail_smooth_width)
                if q_index_90 is not None:
                    pred_q90_all = apply_tail_adjustment(pred_q90_all, era5_centers,
                                                         tail_ratio, tail_threshold, tail_smooth_width)
                if q_index_95 is not None:
                    pred_q95_all = apply_tail_adjustment(pred_q95_all, era5_centers,
                                                         tail_ratio, tail_threshold, tail_smooth_width)

            # 写入 (H,W)
            corrected_mid[t] = pred_mid_all.reshape(H, W).astype(np.float32)
            corrected_q90[t] = pred_q90_all.reshape(H, W).astype(np.float32)
            corrected_q95[t] = pred_q95_all.reshape(H, W).astype(np.float32)
            occ_prob_all[t]  = occ_prob_all_day.reshape(H, W).astype(np.float32)

    # 组装 Dataset
    ds_out = xr.Dataset(
        {
            "corrected_tp":  (("date", "latitude", "longitude"), corrected_mid),
            "corrected_q90": (("date", "latitude", "longitude"), corrected_q90),
            "corrected_q95": (("date", "latitude", "longitude"), corrected_q95),
            "occurrence_prob": (("date", "latitude", "longitude"), occ_prob_all),
            "era5_raw": (("date", "latitude", "longitude"), era5_raw)
        },
        coords={
            "date": dates,
            "latitude": lat,
            "longitude": lon
        },
        attrs={
            "model_path": str(args.model_path),
            "quantiles": ",".join(map(str, quantiles)),
            "point_quantile": float(quantile_mid),
            "occurrence_threshold": float(occ_prob_threshold),
            "drizzle_cut": float(drizzle_cut),
            "tail_adjust_enabled": int(enable_tail),
            "tail_ratio": float(tail_ratio if enable_tail else 1.0),
            "tail_threshold": float(tail_threshold),
            "tail_smooth_width": float(tail_smooth_width)
        }
    )
    ds_out = sanitize_netcdf_attrs(ds_out)
    return ds_out


# ==============================
# 主程序
# ==============================
def main():
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 量化分位 (与训练脚本一致)
    quantiles = (0.5, 0.9, 0.95)

    # 模型加载
    print("[INFO] 加载模型:", args.model_path)
    model = MultiTaskQuantileCNN(in_channels=21, patch_size=args.patch_size, quantiles=quantiles)
    state = torch.load(args.model_path, map_location='cpu')

    # 兼容 {"model": state_dict, ...}
    if isinstance(state, dict) and 'model' in state and all(
        k.startswith(('occ_head', 'resid_head', 'backbone')) for k in state['model'].keys()
    ):
        model.load_state_dict(state['model'])
    else:
        # 直接是 state_dict
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 辅助数据
    print("[INFO] 读取 DEM / Slope ...")
    dem_arr = load_tif(DEM_PATH)
    slope_arr = load_tif(SLOPE_PATH)

    print("[INFO] 读取 Niño 指数 ...")
    try:
        nino_map = load_nino_map(NINO_CSV)
    except Exception as e:
        warnings.warn(f"读取 Niño 失败，将全部置 0: {e}")
        nino_map = {}

    summary_stats = {}

    for year in range(args.year_start, args.year_end + 1):
        print(f"\n======== 处理年份 {year} ========")
        ds_year = correct_year(
            year=year,
            model=model,
            dem_arr=dem_arr,
            slope_arr=slope_arr,
            nino_map=nino_map,
            device=device,
            quantiles=quantiles,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            occ_prob_threshold=args.occ_prob_threshold,
            drizzle_cut=args.drizzle_cut,
            enable_tail=args.enable_tail_adjust,
            tail_ratio=args.tail_ratio,
            tail_threshold=args.tail_threshold,
            tail_smooth_width=args.tail_smooth_width,
            quantile_mid=args.quantile_mid
        )

        out_nc = os.path.join(args.output_dir, f"ERA5_corrected_multitask_{year}.nc")
        ds_year.to_netcdf(out_nc)
        print(f"[INFO] 保存: {out_nc}")

        if args.save_intermediate_json:
            arr_mid = ds_year['corrected_tp'].values
            q_list = [0.5, 0.75, 0.9, 0.95, 0.99]
            stats = {f"q{int(q*100)}": float(np.nanquantile(arr_mid, q)) for q in q_list}
            stats["mean"] = float(np.nanmean(arr_mid))
            summary_stats[year] = stats

    if args.save_intermediate_json:
        summary_path = os.path.join(args.output_dir, "yearly_summary_stats.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 年度统计已保存: {summary_path}")

    print("\n[DONE] 全部年份处理完成。")


if __name__ == "__main__":
    main()