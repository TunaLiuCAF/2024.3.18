"""
降水残差校正数据集类
支持ERA5表面气压作为输入特征
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import netCDF4 as nc
from typing import Tuple, Optional


class PrecipDataset(Dataset):
    """
    降水残差校正数据集
    
    输入特征维度：17 (原16个特征 + ERA5表面气压 Psurf_f_ins)
    """
    
    def __init__(
        self,
        era5_precip_path: str,
        era5_pressure_path: str,
        observation_path: str,
        patch_size: int = 32,
        date_start: str = "2008-01-01",
        date_end: str = "2019-12-31",
        normalize: bool = True
    ):
        """
        初始化数据集
        
        Args:
            era5_precip_path: ERA5降水数据路径
            era5_pressure_path: ERA5表面气压数据路径 (包含Psurf_f_ins变量)
            observation_path: 观测数据路径
            patch_size: patch大小
            date_start: 开始日期
            date_end: 结束日期
            normalize: 是否标准化
        """
        self.era5_precip_path = era5_precip_path
        self.era5_pressure_path = era5_pressure_path
        self.observation_path = observation_path
        self.patch_size = patch_size
        self.date_start = date_start
        self.date_end = date_end
        self.normalize = normalize
        
        # 加载数据
        self._load_data()
        
        # 生成patch索引
        self._generate_patch_indices()
        
        # 计算标准化参数
        if self.normalize:
            self._compute_normalization_stats()
    
    def _load_data(self):
        """加载ERA5和观测数据"""
        print("正在加载ERA5降水数据...")
        with nc.Dataset(self.era5_precip_path, 'r') as f:
            # ERA5降水数据，假设维度为 (time, lat, lon)
            self.era5_precip = f['tp'][:]  # total precipitation
            self.dates = f['time'][:]
            self.lats = f['latitude'][:]
            self.lons = f['longitude'][:]
        
        print("正在加载ERA5表面气压数据...")
        with nc.Dataset(self.era5_pressure_path, 'r') as f:
            # ERA5表面气压数据，维度与降水一致 (time, lat, lon)
            self.era5_pressure = f['Psurf_f_ins'][:]  # 表面气压
        
        print("正在加载观测数据...")
        with nc.Dataset(self.observation_path, 'r') as f:
            # 观测降水数据
            self.obs_precip = f['precipitation'][:]
        
        # 确保数据维度一致
        assert self.era5_precip.shape == self.era5_pressure.shape == self.obs_precip.shape, \
            "ERA5降水、气压和观测数据维度必须一致"
        
        print(f"数据加载完成，shape: {self.era5_precip.shape}")
    
    def _generate_patch_indices(self):
        """生成patch采样索引"""
        self.patch_indices = []
        
        n_time, n_lat, n_lon = self.era5_precip.shape
        
        # 确保patch不超出边界
        for t in range(n_time):
            for i in range(0, n_lat - self.patch_size + 1, self.patch_size // 2):
                for j in range(0, n_lon - self.patch_size + 1, self.patch_size // 2):
                    self.patch_indices.append((t, i, j))
        
        print(f"生成了 {len(self.patch_indices)} 个patch")
    
    def _compute_normalization_stats(self):
        """计算标准化统计参数"""
        print("计算标准化参数...")
        
        # 对所有特征通道计算均值和标准差
        # 包括16个原始特征 + 1个气压特征
        
        # 假设原始16个特征来自多个ERA5变量的组合
        # 这里简化处理，实际应用中需要根据具体特征设计
        all_features = []
        
        # 添加原始ERA5降水特征 (假设经过处理后有16个通道)
        for i in range(16):
            # 这里使用ERA5降水数据的不同变换作为示例
            # 实际应用中应该是不同的气象变量
            feature = self.era5_precip * (i + 1) / 16.0  # 简化示例
            all_features.append(feature)
        
        # 添加表面气压特征作为第17个通道
        all_features.append(self.era5_pressure)
        
        # 堆叠所有特征
        self.features = np.stack(all_features, axis=1)  # (time, 17, lat, lon)
        
        # 计算每个通道的均值和标准差
        self.feature_mean = np.mean(self.features, axis=(0, 2, 3), keepdims=True)
        self.feature_std = np.std(self.features, axis=(0, 2, 3), keepdims=True)
        
        # 避免除零
        self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)
        
        print(f"特征标准化参数计算完成，shape: {self.feature_mean.shape}")
    
    def _extract_patch(self, data: np.ndarray, t: int, i: int, j: int) -> np.ndarray:
        """提取patch"""
        if len(data.shape) == 3:  # (time, lat, lon)
            return data[t, i:i+self.patch_size, j:j+self.patch_size]
        elif len(data.shape) == 4:  # (time, channels, lat, lon)
            return data[t, :, i:i+self.patch_size, j:j+self.patch_size]
        else:
            raise ValueError(f"不支持的数据维度: {data.shape}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.patch_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            input_features: (17, patch_size, patch_size) - 包含气压的17通道输入特征
            target: (patch_size, patch_size) - 目标残差
        """
        t, i, j = self.patch_indices[idx]
        
        # 提取输入特征patch (17通道)
        input_patch = self._extract_patch(self.features, t, i, j)
        
        # 提取目标patch
        era5_patch = self._extract_patch(self.era5_precip, t, i, j)
        obs_patch = self._extract_patch(self.obs_precip, t, i, j)
        
        # 计算残差 (观测 - ERA5)
        residual_patch = obs_patch - era5_patch
        
        # 标准化
        if self.normalize:
            input_patch = (input_patch - self.feature_mean) / self.feature_std
        
        # 转换为tensor
        input_tensor = torch.tensor(input_patch, dtype=torch.float32)
        target_tensor = torch.tensor(residual_patch, dtype=torch.float32)
        
        return input_tensor, target_tensor
    
    def get_feature_names(self) -> list:
        """获取特征名称列表"""
        # 原始16个特征名称 (示例)
        feature_names = [
            'era5_precip_t0', 'era5_precip_t1', 'era5_precip_t2', 'era5_precip_t3',
            'era5_temp_2m', 'era5_temp_850', 'era5_temp_500', 'era5_temp_250',
            'era5_humid_2m', 'era5_humid_850', 'era5_humid_500', 'era5_humid_250',
            'era5_wind_u_10m', 'era5_wind_v_10m', 'era5_wind_u_850', 'era5_wind_v_850'
        ]
        
        # 添加表面气压作为第17个特征
        feature_names.append('Psurf_f_ins')
        
        return feature_names
    
    def get_sample_info(self, idx: int) -> dict:
        """获取样本详细信息"""
        t, i, j = self.patch_indices[idx]
        
        return {
            'time_index': t,
            'lat_start': i,
            'lon_start': j,
            'patch_size': self.patch_size,
            'feature_channels': 17,
            'includes_pressure': True
        }


class PrecipInferenceDataset(Dataset):
    """
    用于推理的降水数据集
    包含ERA5表面气压特征
    """
    
    def __init__(
        self,
        era5_precip_path: str,
        era5_pressure_path: str,
        patch_size: int = 32,
        normalize_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """
        初始化推理数据集
        
        Args:
            era5_precip_path: ERA5降水数据路径
            era5_pressure_path: ERA5表面气压数据路径
            patch_size: patch大小
            normalize_stats: 标准化参数 (mean, std)
        """
        self.era5_precip_path = era5_precip_path
        self.era5_pressure_path = era5_pressure_path
        self.patch_size = patch_size
        self.normalize_stats = normalize_stats
        
        # 加载数据
        self._load_data()
        
        # 生成patch索引
        self._generate_patch_indices()
    
    def _load_data(self):
        """加载数据"""
        print("正在加载推理数据...")
        
        with nc.Dataset(self.era5_precip_path, 'r') as f:
            self.era5_precip = f['tp'][:]
            self.dates = f['time'][:]
            self.lats = f['latitude'][:]
            self.lons = f['longitude'][:]
        
        with nc.Dataset(self.era5_pressure_path, 'r') as f:
            self.era5_pressure = f['Psurf_f_ins'][:]
        
        # 构建17通道特征
        all_features = []
        
        # 原始16个特征 (简化示例)
        for i in range(16):
            feature = self.era5_precip * (i + 1) / 16.0
            all_features.append(feature)
        
        # 添加表面气压
        all_features.append(self.era5_pressure)
        
        self.features = np.stack(all_features, axis=1)  # (time, 17, lat, lon)
        
        print(f"推理数据加载完成，shape: {self.features.shape}")
    
    def _generate_patch_indices(self):
        """生成patch索引"""
        self.patch_indices = []
        
        n_time, _, n_lat, n_lon = self.features.shape
        
        for t in range(n_time):
            for i in range(0, n_lat - self.patch_size + 1, self.patch_size):
                for j in range(0, n_lon - self.patch_size + 1, self.patch_size):
                    self.patch_indices.append((t, i, j))
        
        print(f"生成了 {len(self.patch_indices)} 个推理patch")
    
    def __len__(self) -> int:
        return len(self.patch_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        获取推理样本
        
        Returns:
            input_features: (17, patch_size, patch_size)
            metadata: 包含位置等信息的字典
        """
        t, i, j = self.patch_indices[idx]
        
        # 提取特征patch
        input_patch = self.features[t, :, i:i+self.patch_size, j:j+self.patch_size]
        
        # 标准化
        if self.normalize_stats is not None:
            mean, std = self.normalize_stats
            input_patch = (input_patch - mean) / std
        
        # 转换为tensor
        input_tensor = torch.tensor(input_patch, dtype=torch.float32)
        
        # 元数据
        metadata = {
            'time_index': t,
            'lat_start': i,
            'lon_start': j,
            'patch_size': self.patch_size
        }
        
        return input_tensor, metadata