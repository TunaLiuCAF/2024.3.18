"""
配置文件
降水残差校正模型配置 (支持17通道输入，包含ERA5表面气压)
"""

# 数据配置
DATA_CONFIG = {
    # 输入数据路径
    "era5_precip_path": "/path/to/era5_precipitation.nc",
    "era5_pressure_path": "/path/to/era5_surface_pressure.nc",  # 包含 Psurf_f_ins 变量
    "observation_path": "/path/to/observations.nc",
    
    # 数据处理参数
    "patch_size": 32,
    "date_start": "2008-01-01",
    "date_end": "2019-12-31",
    "normalize": True,
    
    # 数据验证
    "expected_variables": {
        "era5_precip": "tp",  # ERA5降水变量名
        "era5_pressure": "Psurf_f_ins",  # ERA5表面气压变量名
        "observation": "precipitation"  # 观测降水变量名
    }
}

# 模型配置
MODEL_CONFIG = {
    # 基础参数
    "model_type": "standard",  # "standard" 或 "lightweight"
    "input_channels": 17,  # 17个通道：16个原始特征 + 1个表面气压
    "output_channels": 1,  # 输出残差
    
    # 网络结构
    "hidden_channels": [64, 128, 256, 512],
    "use_attention": True,
    "dropout": 0.1,
    
    # 特征配置
    "feature_names": [
        "era5_precip_t0", "era5_precip_t1", "era5_precip_t2", "era5_precip_t3",
        "era5_temp_2m", "era5_temp_850", "era5_temp_500", "era5_temp_250",
        "era5_humid_2m", "era5_humid_850", "era5_humid_500", "era5_humid_250",
        "era5_wind_u_10m", "era5_wind_v_10m", "era5_wind_u_850", "era5_wind_v_850",
        "Psurf_f_ins"  # 第17个特征：ERA5表面气压
    ],
    
    # 特征重要性权重 (可选)
    "feature_weights": None,  # 如果为None，所有特征权重相等
    
    # 表面气压特征配置
    "pressure_config": {
        "channel_index": 16,  # 气压在第17个通道 (索引16)
        "normalization": "standard",  # "standard", "minmax", 或 "none"
        "pressure_range": [950, 1050],  # 预期气压范围 (hPa)
        "feature_enhancement": True  # 是否对气压特征进行增强
    }
}

# 训练配置
TRAIN_CONFIG = {
    # 数据分割
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "random_seed": 42,
    
    # 训练参数
    "batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    
    # 优化器配置
    "optimizer": "adamw",  # "adam", "adamw", "sgd"
    "momentum": 0.9,  # 仅对SGD有效
    "betas": (0.9, 0.999),  # 仅对Adam/AdamW有效
    
    # 学习率调度
    "scheduler": "cosine",  # "step", "cosine", "reduce", None
    "step_size": 20,  # 仅对StepLR有效
    "gamma": 0.1,  # 学习率衰减因子
    "T_max": 100,  # 仅对CosineAnnealingLR有效
    "patience": 10,  # 仅对ReduceLROnPlateau有效
    "factor": 0.5,  # 仅对ReduceLROnPlateau有效
    
    # 损失函数
    "loss_type": "combined",  # "mse", "mae", "huber", "combined"
    "loss_weights": {
        "mse_weight": 0.7,
        "mae_weight": 0.3
    },
    
    # 正则化
    "grad_clip": 1.0,  # 梯度裁剪
    "dropout": 0.1,
    
    # 数据加载
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    
    # 保存和恢复
    "save_interval": 10,  # 每N个epoch保存一次
    "save_best_only": False,
    "early_stopping_patience": 20,
    
    # 验证和监控
    "validation_interval": 1,  # 每N个epoch验证一次
    "log_interval": 100,  # 每N个batch记录一次
    
    # 增强气压特征训练
    "pressure_training": {
        "enable": True,
        "warmup_epochs": 10,  # 前N个epoch专注训练气压特征
        "pressure_loss_weight": 1.5,  # 气压相关损失的权重
        "progressive_training": True  # 渐进式训练
    }
}

# 推理配置
INFERENCE_CONFIG = {
    # 推理参数
    "batch_size": 32,
    "patch_size": 32,
    "overlap": 0.5,  # patch重叠比例
    
    # 输出配置
    "save_residual": True,
    "save_corrected": True,
    "save_confidence": False,  # 是否保存置信度
    
    # 后处理
    "post_processing": {
        "ensure_non_negative": True,  # 确保降水非负
        "smooth_boundaries": True,  # 平滑patch边界
        "apply_mask": False,  # 是否应用掩膜
        "mask_threshold": 0.01  # 掩膜阈值
    },
    
    # 并行处理
    "num_workers": 4,
    "use_gpu": True,
    "gpu_memory_limit": 0.8,  # GPU内存使用限制
    
    # 输出格式
    "output_format": "netcdf",  # "netcdf", "tiff", "numpy"
    "compression": {
        "enable": True,
        "level": 4,
        "algorithm": "zlib"
    }
}

# 评估配置
EVALUATION_CONFIG = {
    # 评估指标
    "metrics": [
        "rmse", "mae", "correlation", "bias",
        "skill_score", "nash_sutcliffe", "kge"
    ],
    
    # 空间评估
    "spatial_evaluation": {
        "enable": True,
        "grid_size": [5, 5],  # 将区域划分为5x5网格评估
        "percentile_analysis": [50, 75, 90, 95, 99]
    },
    
    # 时间评估
    "temporal_evaluation": {
        "enable": True,
        "monthly_analysis": True,
        "seasonal_analysis": True,
        "extreme_event_analysis": True
    },
    
    # 特征重要性分析
    "feature_importance": {
        "enable": True,
        "method": "permutation",  # "permutation", "shap", "gradient"
        "analyze_pressure": True  # 专门分析气压特征的重要性
    },
    
    # 可视化
    "visualization": {
        "enable": True,
        "save_plots": True,
        "plot_format": "png",
        "dpi": 300,
        "include_pressure_analysis": True  # 包含气压相关分析图
    }
}

# 实验配置
EXPERIMENT_CONFIG = {
    # 实验信息
    "experiment_name": "precip_correction_with_pressure",
    "description": "降水残差校正，添加ERA5表面气压作为第17个输入特征",
    "version": "1.0",
    "author": "Precipitation Correction Team",
    
    # 对比实验
    "ablation_study": {
        "enable": True,
        "experiments": [
            {
                "name": "baseline_16_channels",
                "description": "基线模型，仅使用16个原始特征",
                "input_channels": 16,
                "exclude_pressure": True
            },
            {
                "name": "with_pressure_17_channels",
                "description": "完整模型，包含17个特征（含表面气压）",
                "input_channels": 17,
                "include_pressure": True
            }
        ]
    },
    
    # 目录配置
    "directories": {
        "data_dir": "./data",
        "model_dir": "./models",
        "output_dir": "./outputs",
        "log_dir": "./logs",
        "checkpoint_dir": "./checkpoints",
        "visualization_dir": "./visualizations"
    },
    
    # 资源配置
    "resources": {
        "max_memory_gb": 32,
        "max_gpu_memory_gb": 12,
        "num_cpus": 8,
        "storage_limit_gb": 100
    }
}

# 完整配置字典
CONFIG = {
    "data": DATA_CONFIG,
    "model": MODEL_CONFIG,
    "train": TRAIN_CONFIG,
    "inference": INFERENCE_CONFIG,
    "evaluation": EVALUATION_CONFIG,
    "experiment": EXPERIMENT_CONFIG
}

# 配置验证函数
def validate_config(config=None):
    """验证配置的有效性"""
    if config is None:
        config = CONFIG
    
    errors = []
    
    # 验证输入通道数
    if config["model"]["input_channels"] != 17:
        errors.append("模型输入通道数必须为17（包含表面气压）")
    
    # 验证特征名称数量
    if len(config["model"]["feature_names"]) != 17:
        errors.append("特征名称列表必须包含17个特征")
    
    # 验证气压特征配置
    if "Psurf_f_ins" not in config["model"]["feature_names"]:
        errors.append("特征列表必须包含 'Psurf_f_ins'")
    
    # 验证数据路径
    required_paths = ["era5_precip_path", "era5_pressure_path", "observation_path"]
    for path_key in required_paths:
        if not config["data"].get(path_key):
            errors.append(f"必须设置 {path_key}")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(f"- {error}" for error in errors))
    
    print("✓ 配置验证通过")
    return True

# 获取配置函数
def get_config(config_type=None):
    """获取特定类型的配置"""
    if config_type is None:
        return CONFIG
    elif config_type in CONFIG:
        return CONFIG[config_type]
    else:
        raise ValueError(f"未知的配置类型: {config_type}")

# 更新配置函数
def update_config(updates, config_type=None):
    """更新配置"""
    if config_type is None:
        target_config = CONFIG
    else:
        target_config = CONFIG[config_type]
    
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(target_config, updates)
    return target_config


if __name__ == "__main__":
    # 测试配置
    print("=== 配置测试 ===")
    
    try:
        # 验证配置
        validate_config()
        
        # 打印关键配置信息
        print(f"输入通道数: {CONFIG['model']['input_channels']}")
        print(f"特征数量: {len(CONFIG['model']['feature_names'])}")
        print(f"包含表面气压: {'Psurf_f_ins' in CONFIG['model']['feature_names']}")
        print(f"气压特征索引: {CONFIG['model']['pressure_config']['channel_index']}")
        
        print("✓ 配置测试通过")
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")