"""
降水残差校正推理流程
支持ERA5表面气压 (Psurf_f_ins) 作为输入特征
"""

import torch
import numpy as np
import netCDF4 as nc
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import argparse
from torch.utils.data import DataLoader

from dataset import PrecipInferenceDataset
from model import create_model


class PrecipCorrectionInference:
    """降水残差校正推理器"""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None
    ):
        """
        初始化推理器
        
        Args:
            model_path: 训练好的模型路径
            device: 推理设备
        """
        self.model_path = Path(model_path)
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载训练好的模型"""
        print(f"正在加载模型: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取模型配置
        model_config = checkpoint['model_config']
        print(f"模型配置: {model_config}")
        
        # 创建模型
        self.model = create_model(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 获取标准化参数
        if 'normalization_stats' in checkpoint:
            self.normalize_stats = checkpoint['normalization_stats']
        else:
            self.normalize_stats = None
            print("警告: 模型中没有标准化参数，将使用原始数据")
        
        print("模型加载完成！")
        
        # 打印模型信息
        model_info = self.model.get_model_info()
        print(f"模型信息: {model_info}")
        
        if model_info.get('includes_pressure', False):
            print("✓ 模型支持ERA5表面气压特征")
        else:
            print("✗ 模型不包含表面气压特征")
    
    def correct_precipitation(
        self,
        era5_precip_path: str,
        era5_pressure_path: str,
        output_path: str,
        patch_size: int = 32,
        batch_size: int = 16,
        save_residual: bool = True
    ) -> Dict:
        """
        执行降水校正
        
        Args:
            era5_precip_path: ERA5降水数据路径
            era5_pressure_path: ERA5表面气压数据路径 (包含Psurf_f_ins)
            output_path: 输出文件路径
            patch_size: patch大小
            batch_size: 批次大小
            save_residual: 是否保存残差
        
        Returns:
            校正结果统计信息
        """
        print("开始降水校正...")
        start_time = time.time()
        
        # 创建推理数据集
        inference_dataset = PrecipInferenceDataset(
            era5_precip_path=era5_precip_path,
            era5_pressure_path=era5_pressure_path,
            patch_size=patch_size,
            normalize_stats=self.normalize_stats
        )
        
        # 创建数据加载器
        inference_loader = DataLoader(
            inference_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=4
        )
        
        # 获取原始数据维度
        with nc.Dataset(era5_precip_path, 'r') as f:
            original_shape = f['tp'].shape
            dates = f['time'][:]
            lats = f['latitude'][:]
            lons = f['longitude'][:]
        
        print(f"原始数据shape: {original_shape}")
        
        # 初始化结果数组
        corrected_precip = np.zeros_like(inference_dataset.era5_precip)
        residual_field = np.zeros_like(inference_dataset.era5_precip)
        
        # 推理
        print("正在进行推理...")
        with torch.no_grad():
            for batch_idx, (inputs, metadata_list) in enumerate(inference_loader):
                # 移到设备
                inputs = inputs.to(self.device)  # (batch_size, 17, patch_size, patch_size)
                
                # 模型推理
                predicted_residuals = self.model(inputs)  # (batch_size, 1, patch_size, patch_size)
                
                # 转换为numpy
                predicted_residuals = predicted_residuals.squeeze(1).cpu().numpy()
                
                # 将结果放回对应位置
                for i, metadata in enumerate(metadata_list):
                    t = metadata['time_index']
                    lat_start = metadata['lat_start']
                    lon_start = metadata['lon_start']
                    
                    # 提取当前patch的残差
                    residual_patch = predicted_residuals[i]
                    
                    # 放回结果数组
                    residual_field[
                        t, 
                        lat_start:lat_start+patch_size, 
                        lon_start:lon_start+patch_size
                    ] = residual_patch
                
                if batch_idx % 100 == 0:
                    print(f"处理批次: {batch_idx}/{len(inference_loader)}")
        
        # 应用校正
        print("正在应用校正...")
        corrected_precip = inference_dataset.era5_precip + residual_field
        
        # 确保降水非负
        corrected_precip = np.maximum(corrected_precip, 0)
        
        # 保存结果
        self._save_results(
            output_path=output_path,
            corrected_precip=corrected_precip,
            residual_field=residual_field,
            era5_precip=inference_dataset.era5_precip,
            dates=dates,
            lats=lats,
            lons=lons,
            save_residual=save_residual
        )
        
        # 计算统计信息
        stats = self._compute_statistics(
            era5_precip=inference_dataset.era5_precip,
            corrected_precip=corrected_precip,
            residual_field=residual_field
        )
        
        total_time = time.time() - start_time
        print(f"校正完成！用时: {total_time:.2f}秒")
        
        return stats
    
    def _save_results(
        self,
        output_path: str,
        corrected_precip: np.ndarray,
        residual_field: np.ndarray,
        era5_precip: np.ndarray,
        dates: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        save_residual: bool = True
    ):
        """保存校正结果"""
        print(f"正在保存结果到: {output_path}")
        
        with nc.Dataset(output_path, 'w') as f:
            # 创建维度
            f.createDimension('time', len(dates))
            f.createDimension('latitude', len(lats))
            f.createDimension('longitude', len(lons))
            
            # 创建坐标变量
            time_var = f.createVariable('time', 'f8', ('time',))
            lat_var = f.createVariable('latitude', 'f4', ('latitude',))
            lon_var = f.createVariable('longitude', 'f4', ('longitude',))
            
            time_var[:] = dates
            lat_var[:] = lats
            lon_var[:] = lons
            
            # 设置坐标属性
            time_var.units = 'hours since 1900-01-01 00:00:0.0'
            lat_var.units = 'degrees_north'
            lon_var.units = 'degrees_east'
            
            # 创建降水变量
            # 原始ERA5降水
            era5_var = f.createVariable(
                'era5_precipitation', 'f4', ('time', 'latitude', 'longitude'),
                zlib=True, complevel=4, fill_value=-9999
            )
            era5_var[:] = era5_precip
            era5_var.units = 'mm'
            era5_var.long_name = 'ERA5 Original Precipitation'
            
            # 校正后降水
            corrected_var = f.createVariable(
                'corrected_precipitation', 'f4', ('time', 'latitude', 'longitude'),
                zlib=True, complevel=4, fill_value=-9999
            )
            corrected_var[:] = corrected_precip
            corrected_var.units = 'mm'
            corrected_var.long_name = 'Corrected Precipitation with Surface Pressure'
            
            # 残差场 (可选)
            if save_residual:
                residual_var = f.createVariable(
                    'residual_field', 'f4', ('time', 'latitude', 'longitude'),
                    zlib=True, complevel=4, fill_value=-9999
                )
                residual_var[:] = residual_field
                residual_var.units = 'mm'
                residual_var.long_name = 'Predicted Residual Field'
            
            # 全局属性
            f.title = 'Precipitation Residual Correction with ERA5 Surface Pressure'
            f.institution = 'Precipitation Correction System'
            f.source = 'Residual Correction Model with 17 input channels (including Psurf_f_ins)'
            f.history = f'Created on {time.strftime("%Y-%m-%d %H:%M:%S")}'
            f.description = 'Corrected precipitation using ERA5 surface pressure as additional input feature'
        
        print("结果保存完成！")
    
    def _compute_statistics(
        self,
        era5_precip: np.ndarray,
        corrected_precip: np.ndarray,
        residual_field: np.ndarray
    ) -> Dict:
        """计算校正统计信息"""
        print("正在计算统计信息...")
        
        # 基本统计
        stats = {
            'era5_stats': {
                'mean': float(np.mean(era5_precip)),
                'std': float(np.std(era5_precip)),
                'min': float(np.min(era5_precip)),
                'max': float(np.max(era5_precip)),
                'total': float(np.sum(era5_precip))
            },
            'corrected_stats': {
                'mean': float(np.mean(corrected_precip)),
                'std': float(np.std(corrected_precip)),
                'min': float(np.min(corrected_precip)),
                'max': float(np.max(corrected_precip)),
                'total': float(np.sum(corrected_precip))
            },
            'residual_stats': {
                'mean': float(np.mean(residual_field)),
                'std': float(np.std(residual_field)),
                'min': float(np.min(residual_field)),
                'max': float(np.max(residual_field)),
                'abs_mean': float(np.mean(np.abs(residual_field)))
            }
        }
        
        # 校正改善比例
        correction_magnitude = np.abs(residual_field)
        positive_corrections = np.sum(residual_field > 0)
        negative_corrections = np.sum(residual_field < 0)
        total_points = residual_field.size
        
        stats['correction_analysis'] = {
            'mean_correction_magnitude': float(np.mean(correction_magnitude)),
            'positive_corrections_ratio': float(positive_corrections / total_points),
            'negative_corrections_ratio': float(negative_corrections / total_points),
            'significant_corrections_ratio': float(
                np.sum(correction_magnitude > 0.1) / total_points
            )
        }
        
        return stats
    
    def evaluate_with_observations(
        self,
        corrected_precip_path: str,
        observation_path: str,
        era5_precip_path: str
    ) -> Dict:
        """
        使用观测数据评估校正效果
        
        Args:
            corrected_precip_path: 校正后降水数据路径
            observation_path: 观测数据路径
            era5_precip_path: 原始ERA5数据路径
        
        Returns:
            评估结果
        """
        print("正在评估校正效果...")
        
        # 加载数据
        with nc.Dataset(corrected_precip_path, 'r') as f:
            corrected_precip = f['corrected_precipitation'][:]
        
        with nc.Dataset(observation_path, 'r') as f:
            observations = f['precipitation'][:]
        
        with nc.Dataset(era5_precip_path, 'r') as f:
            era5_precip = f['tp'][:]
        
        # 确保维度一致
        assert corrected_precip.shape == observations.shape == era5_precip.shape
        
        # 计算评估指标
        # 均方根误差 (RMSE)
        rmse_era5 = np.sqrt(np.mean((era5_precip - observations) ** 2))
        rmse_corrected = np.sqrt(np.mean((corrected_precip - observations) ** 2))
        
        # 平均绝对误差 (MAE)
        mae_era5 = np.mean(np.abs(era5_precip - observations))
        mae_corrected = np.mean(np.abs(corrected_precip - observations))
        
        # 相关系数
        corr_era5 = np.corrcoef(era5_precip.flatten(), observations.flatten())[0, 1]
        corr_corrected = np.corrcoef(corrected_precip.flatten(), observations.flatten())[0, 1]
        
        # 偏差 (Bias)
        bias_era5 = np.mean(era5_precip - observations)
        bias_corrected = np.mean(corrected_precip - observations)
        
        evaluation_results = {
            'era5_metrics': {
                'rmse': float(rmse_era5),
                'mae': float(mae_era5),
                'correlation': float(corr_era5),
                'bias': float(bias_era5)
            },
            'corrected_metrics': {
                'rmse': float(rmse_corrected),
                'mae': float(mae_corrected),
                'correlation': float(corr_corrected),
                'bias': float(bias_corrected)
            },
            'improvement': {
                'rmse_improvement': float((rmse_era5 - rmse_corrected) / rmse_era5 * 100),
                'mae_improvement': float((mae_era5 - mae_corrected) / mae_era5 * 100),
                'correlation_improvement': float(corr_corrected - corr_era5),
                'bias_reduction': float(abs(bias_era5) - abs(bias_corrected))
            }
        }
        
        print("评估完成！")
        print(f"RMSE改善: {evaluation_results['improvement']['rmse_improvement']:.2f}%")
        print(f"MAE改善: {evaluation_results['improvement']['mae_improvement']:.2f}%")
        print(f"相关性改善: {evaluation_results['improvement']['correlation_improvement']:.4f}")
        
        return evaluation_results


def main():
    """主函数 - 推理示例"""
    parser = argparse.ArgumentParser(description='降水残差校正推理')
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--era5_precip', required=True, help='ERA5降水数据路径')
    parser.add_argument('--era5_pressure', required=True, help='ERA5表面气压数据路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    parser.add_argument('--observation', help='观测数据路径 (用于评估)')
    parser.add_argument('--patch_size', type=int, default=32, help='patch大小')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', help='推理设备')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = PrecipCorrectionInference(
        model_path=args.model_path,
        device=args.device
    )
    
    # 执行校正
    stats = inference.correct_precipitation(
        era5_precip_path=args.era5_precip,
        era5_pressure_path=args.era5_pressure,
        output_path=args.output,
        patch_size=args.patch_size,
        batch_size=args.batch_size
    )
    
    # 保存统计信息
    stats_path = Path(args.output).parent / 'correction_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"校正统计信息已保存到: {stats_path}")
    
    # 如果提供了观测数据，进行评估
    if args.observation:
        evaluation = inference.evaluate_with_observations(
            corrected_precip_path=args.output,
            observation_path=args.observation,
            era5_precip_path=args.era5_precip
        )
        
        # 保存评估结果
        eval_path = Path(args.output).parent / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        print(f"评估结果已保存到: {eval_path}")


if __name__ == "__main__":
    main()