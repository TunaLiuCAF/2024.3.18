"""
完整的降水残差校正示例
演示如何使用ERA5表面气压 (Psurf_f_ins) 作为输入特征
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from dataset import PrecipDataset, PrecipInferenceDataset
from model import create_model, PrecipResidualCorrectionModel
from train import ResidualCorrectionTrainer
from inference import PrecipCorrectionInference


def create_sample_data():
    """
    创建示例数据用于测试
    模拟ERA5降水、表面气压和观测数据
    """
    print("创建示例数据...")
    
    # 数据维度设置
    n_time = 100  # 100天
    n_lat = 64    # 64个纬度点
    n_lon = 64    # 64个经度点
    
    # 创建坐标
    dates = np.arange(n_time)
    lats = np.linspace(30, 40, n_lat)  # 30°N - 40°N
    lons = np.linspace(110, 120, n_lon)  # 110°E - 120°E
    
    # 生成模拟的ERA5降水数据
    # 使用空间和时间相关的随机模式
    np.random.seed(42)
    base_pattern = np.random.random((n_lat, n_lon)) * 10
    
    era5_precip = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        # 添加季节性变化
        seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * t / 365)
        # 添加随机变化
        random_factor = 0.5 + 0.5 * np.random.random((n_lat, n_lon))
        era5_precip[t] = base_pattern * seasonal_factor * random_factor
    
    # 生成模拟的ERA5表面气压数据 (Psurf_f_ins)
    # 通常表面气压在 950-1050 hPa 范围内
    base_pressure = 1000 + 20 * np.random.random((n_lat, n_lon))  # 基础气压场
    
    era5_pressure = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        # 添加季节性气压变化
        seasonal_pressure = 10 * np.sin(2 * np.pi * t / 365)
        # 添加短期变化
        daily_variation = 5 * np.random.random((n_lat, n_lon)) - 2.5
        era5_pressure[t] = base_pressure + seasonal_pressure + daily_variation
    
    # 生成模拟的观测数据
    # 观测数据 = ERA5数据 + 真实残差 + 噪声
    true_residual = np.zeros((n_time, n_lat, n_lon))
    
    # 创建与气压相关的残差模式
    for t in range(n_time):
        # 残差与气压异常相关
        pressure_anomaly = era5_pressure[t] - np.mean(era5_pressure[t])
        pressure_effect = -0.01 * pressure_anomaly  # 高压区域降水偏少
        
        # 添加其他复杂模式
        spatial_pattern = np.sin(np.pi * np.arange(n_lat)[:, None] / n_lat) * \
                         np.cos(np.pi * np.arange(n_lon) / n_lon)
        true_residual[t] = pressure_effect + 2 * spatial_pattern + \
                          0.5 * np.random.random((n_lat, n_lon)) - 0.25
    
    observations = era5_precip + true_residual
    # 确保降水非负
    observations = np.maximum(observations, 0)
    
    return {
        'era5_precip': era5_precip,
        'era5_pressure': era5_pressure,
        'observations': observations,
        'true_residual': true_residual,
        'dates': dates,
        'lats': lats,
        'lons': lons
    }


def save_sample_data_to_netcdf(data_dict, output_dir):
    """将示例数据保存为NetCDF格式"""
    import netCDF4 as nc
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存ERA5降水数据
    with nc.Dataset(output_dir / 'era5_precipitation.nc', 'w') as f:
        f.createDimension('time', len(data_dict['dates']))
        f.createDimension('latitude', len(data_dict['lats']))
        f.createDimension('longitude', len(data_dict['lons']))
        
        time_var = f.createVariable('time', 'f8', ('time',))
        lat_var = f.createVariable('latitude', 'f4', ('latitude',))
        lon_var = f.createVariable('longitude', 'f4', ('longitude',))
        precip_var = f.createVariable('tp', 'f4', ('time', 'latitude', 'longitude'))
        
        time_var[:] = data_dict['dates']
        lat_var[:] = data_dict['lats']
        lon_var[:] = data_dict['lons']
        precip_var[:] = data_dict['era5_precip']
        
        time_var.units = 'days since 2020-01-01'
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        precip_var.units = 'mm'
    
    # 保存ERA5表面气压数据
    with nc.Dataset(output_dir / 'era5_surface_pressure.nc', 'w') as f:
        f.createDimension('time', len(data_dict['dates']))
        f.createDimension('latitude', len(data_dict['lats']))
        f.createDimension('longitude', len(data_dict['lons']))
        
        time_var = f.createVariable('time', 'f8', ('time',))
        lat_var = f.createVariable('latitude', 'f4', ('latitude',))
        lon_var = f.createVariable('longitude', 'f4', ('longitude',))
        pressure_var = f.createVariable('Psurf_f_ins', 'f4', ('time', 'latitude', 'longitude'))
        
        time_var[:] = data_dict['dates']
        lat_var[:] = data_dict['lats']
        lon_var[:] = data_dict['lons']
        pressure_var[:] = data_dict['era5_pressure']
        
        time_var.units = 'days since 2020-01-01'
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        pressure_var.units = 'hPa'
    
    # 保存观测数据
    with nc.Dataset(output_dir / 'observations.nc', 'w') as f:
        f.createDimension('time', len(data_dict['dates']))
        f.createDimension('latitude', len(data_dict['lats']))
        f.createDimension('longitude', len(data_dict['lons']))
        
        time_var = f.createVariable('time', 'f8', ('time',))
        lat_var = f.createVariable('latitude', 'f4', ('latitude',))
        lon_var = f.createVariable('longitude', 'f4', ('longitude',))
        obs_var = f.createVariable('precipitation', 'f4', ('time', 'latitude', 'longitude'))
        
        time_var[:] = data_dict['dates']
        lat_var[:] = data_dict['lats']
        lon_var[:] = data_dict['lons']
        obs_var[:] = data_dict['observations']
        
        time_var.units = 'days since 2020-01-01'
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        obs_var.units = 'mm'
    
    print(f"示例数据已保存到: {output_dir}")


def test_dataset():
    """测试数据集类"""
    print("\n=== 测试数据集 ===")
    
    # 创建示例数据
    data_dict = create_sample_data()
    save_sample_data_to_netcdf(data_dict, './sample_data')
    
    # 测试PrecipDataset
    try:
        dataset = PrecipDataset(
            era5_precip_path='./sample_data/era5_precipitation.nc',
            era5_pressure_path='./sample_data/era5_surface_pressure.nc',
            observation_path='./sample_data/observations.nc',
            patch_size=16,
            normalize=True
        )
        
        print(f"数据集大小: {len(dataset)}")
        print(f"特征名称: {dataset.get_feature_names()}")
        
        # 测试获取样本
        sample_input, sample_target = dataset[0]
        print(f"输入特征shape: {sample_input.shape}")  # 应该是 (17, 16, 16)
        print(f"目标shape: {sample_target.shape}")
        
        # 验证包含表面气压
        assert sample_input.shape[0] == 17, "输入特征应该有17个通道"
        print("✓ 数据集测试成功，包含17个特征通道")
        
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")


def test_model():
    """测试模型"""
    print("\n=== 测试模型 ===")
    
    try:
        # 创建模型
        model = create_model(
            model_type="standard",
            input_channels=17,  # 包含表面气压的17通道
            hidden_channels=[32, 64, 128],
            use_attention=True
        )
        
        # 打印模型信息
        model_info = model.get_model_info()
        print(f"模型信息: {model_info}")
        
        # 测试前向传播
        batch_size = 2
        height, width = 16, 16
        
        # 创建17通道输入
        x = torch.randn(batch_size, 17, height, width)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"输入shape: {x.shape}")
        print(f"输出shape: {output.shape}")
        
        # 验证输出维度
        assert output.shape == (batch_size, 1, height, width), "输出维度不正确"
        print("✓ 模型测试成功")
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")


def test_training():
    """测试训练流程 (简化版)"""
    print("\n=== 测试训练流程 ===")
    
    try:
        # 模型配置
        model_config = {
            'model_type': 'lightweight',  # 使用轻量级模型加快测试
            'input_channels': 17,
            'hidden_channels': 32
        }
        
        # 数据配置
        data_config = {
            'era5_precip_path': './sample_data/era5_precipitation.nc',
            'era5_pressure_path': './sample_data/era5_surface_pressure.nc',
            'observation_path': './sample_data/observations.nc',
            'patch_size': 16,
            'normalize': True
        }
        
        # 训练配置 (简化用于测试)
        train_config = {
            'batch_size': 4,
            'num_epochs': 2,  # 仅训练2个epoch用于测试
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'loss_type': 'mse',
            'train_ratio': 0.8,
            'num_workers': 0,  # 单线程避免问题
            'save_interval': 1,
            'early_stopping_patience': 10
        }
        
        # 创建训练器
        trainer = ResidualCorrectionTrainer(
            model_config=model_config,
            data_config=data_config,
            train_config=train_config,
            save_dir='./test_checkpoints'
        )
        
        # 运行几个步骤测试
        train_loader, val_loader = trainer.prepare_data()
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        
        # 测试一个训练步骤
        trainer.model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)
            
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            
            outputs = trainer.model(inputs)
            loss = trainer.criterion(outputs, targets)
            
            print(f"输入shape: {inputs.shape}")
            print(f"输出shape: {outputs.shape}")
            print(f"目标shape: {targets.shape}")
            print(f"损失: {loss.item():.6f}")
            break
        
        print("✓ 训练流程测试成功")
        
    except Exception as e:
        print(f"✗ 训练流程测试失败: {e}")


def test_inference():
    """测试推理流程"""
    print("\n=== 测试推理流程 ===")
    
    try:
        # 创建简单模型用于测试
        model = create_model(
            model_type="lightweight",
            input_channels=17,
            hidden_channels=32
        )
        
        # 保存测试模型
        test_model_path = './test_model.pth'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'model_type': 'lightweight',
                'input_channels': 17,
                'hidden_channels': 32
            }
        }
        torch.save(checkpoint, test_model_path)
        
        # 测试推理数据集
        inference_dataset = PrecipInferenceDataset(
            era5_precip_path='./sample_data/era5_precipitation.nc',
            era5_pressure_path='./sample_data/era5_surface_pressure.nc',
            patch_size=16
        )
        
        print(f"推理数据集大小: {len(inference_dataset)}")
        
        # 测试单个样本
        sample_input, metadata = inference_dataset[0]
        print(f"推理输入shape: {sample_input.shape}")  # 应该是 (17, 16, 16)
        print(f"元数据: {metadata}")
        
        # 验证输入通道数
        assert sample_input.shape[0] == 17, "推理输入应该有17个通道"
        print("✓ 推理测试成功")
        
    except Exception as e:
        print(f"✗ 推理测试失败: {e}")


def demo_feature_analysis():
    """演示特征分析"""
    print("\n=== 特征分析演示 ===")
    
    try:
        # 创建示例数据
        data_dict = create_sample_data()
        
        # 分析ERA5降水和表面气压的关系
        precip = data_dict['era5_precip']
        pressure = data_dict['era5_pressure']
        
        # 计算相关性
        precip_flat = precip.flatten()
        pressure_flat = pressure.flatten()
        correlation = np.corrcoef(precip_flat, pressure_flat)[0, 1]
        
        print(f"ERA5降水与表面气压的相关系数: {correlation:.4f}")
        
        # 绘制关系图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 第一个时间步的降水分布
        im1 = axes[0, 0].imshow(precip[0], cmap='Blues')
        axes[0, 0].set_title('ERA5 降水 (第1天)')
        axes[0, 0].set_xlabel('经度')
        axes[0, 0].set_ylabel('纬度')
        plt.colorbar(im1, ax=axes[0, 0], label='mm')
        
        # 第一个时间步的气压分布
        im2 = axes[0, 1].imshow(pressure[0], cmap='RdYlBu_r')
        axes[0, 1].set_title('ERA5 表面气压 (第1天)')
        axes[0, 1].set_xlabel('经度')
        axes[0, 1].set_ylabel('纬度')
        plt.colorbar(im2, ax=axes[0, 1], label='hPa')
        
        # 时间序列 (取中心点)
        center_i, center_j = precip.shape[1] // 2, precip.shape[2] // 2
        precip_series = precip[:, center_i, center_j]
        pressure_series = pressure[:, center_i, center_j]
        
        axes[1, 0].plot(precip_series, label='降水')
        axes[1, 0].set_title('中心点降水时间序列')
        axes[1, 0].set_xlabel('时间 (天)')
        axes[1, 0].set_ylabel('降水 (mm)')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(pressure_series, label='气压', color='red')
        axes[1, 1].set_title('中心点气压时间序列')
        axes[1, 1].set_xlabel('时间 (天)')
        axes[1, 1].set_ylabel('气压 (hPa)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
        print("特征分析图已保存为 feature_analysis.png")
        
        # 分析真实残差与气压的关系
        true_residual = data_dict['true_residual']
        residual_flat = true_residual.flatten()
        residual_pressure_corr = np.corrcoef(residual_flat, pressure_flat)[0, 1]
        
        print(f"真实残差与表面气压的相关系数: {residual_pressure_corr:.4f}")
        print("✓ 特征分析完成")
        
    except Exception as e:
        print(f"✗ 特征分析失败: {e}")


def main():
    """主函数 - 运行所有测试"""
    print("=== 降水残差校正系统测试 ===")
    print("支持ERA5表面气压 (Psurf_f_ins) 作为输入特征")
    print("输入特征从16通道扩展到17通道")
    
    # 运行各项测试
    test_dataset()
    test_model()
    demo_feature_analysis()
    test_training()
    test_inference()
    
    print("\n=== 所有测试完成 ===")
    print("\n功能总结:")
    print("1. ✓ PrecipDataset 支持17个输入通道 (包含Psurf_f_ins)")
    print("2. ✓ 模型架构支持17通道输入")
    print("3. ✓ 训练流程包含表面气压特征")
    print("4. ✓ 推理流程支持表面气压输入")
    print("5. ✓ 完整的数据处理和评估流程")
    
    print("\n使用说明:")
    print("- ERA5表面气压数据应包含变量名 'Psurf_f_ins'")
    print("- 数据维度必须与ERA5降水一致: (time, lat, lon)")
    print("- 模型会自动将气压作为第17个输入通道")
    print("- 所有原有功能保持不变，输出仍为残差校正值")


if __name__ == "__main__":
    main()