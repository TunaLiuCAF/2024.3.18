# 降水残差校正模型 - 支持ERA5表面气压特征

本项目实现了一个深度学习降水残差校正模型，将ERA5表面气压（`Psurf_f_ins`）作为新的输入特征添加到模型中，将输入特征维度从16个扩展到17个。

## 项目概述

### 主要功能
- **17通道输入**: 原始16个特征 + ERA5表面气压（`Psurf_f_ins`）
- **残差校正**: 基于深度学习的降水残差预测和校正
- **完整流程**: 数据加载、模型训练、推理校正、结果评估
- **灵活配置**: 支持多种模型架构和训练策略

### 关键改进
1. 添加ERA5表面气压作为第17个输入特征
2. 修改`PrecipDataset`以支持17通道输入
3. 更新模型架构以处理17个输入通道
4. 保持所有原有功能和流程不变

## 文件结构

```
├── dataset.py          # 数据集类（支持17通道）
├── model.py           # 模型定义（17通道输入）
├── train.py           # 训练流程
├── inference.py       # 推理/校正流程
├── config.py          # 配置文件
├── example.py         # 完整示例和测试
└── README.md          # 本文档
```

## 数据要求

### 输入数据格式
所有数据文件都应为NetCDF格式，具有相同的维度结构：`(time, lat, lon)`

1. **ERA5降水数据** (`era5_precipitation.nc`)
   - 变量名: `tp` (total precipitation)
   - 维度: `(time, latitude, longitude)`
   - 单位: mm

2. **ERA5表面气压数据** (`era5_surface_pressure.nc`) ⭐ **新增**
   - 变量名: `Psurf_f_ins` 
   - 维度: `(time, latitude, longitude)` - 与降水数据一致
   - 单位: hPa (百帕)

3. **观测降水数据** (`observations.nc`)
   - 变量名: `precipitation`
   - 维度: `(time, latitude, longitude)`
   - 单位: mm

### 数据质量要求
- 所有数据的时空维度必须完全一致
- 时间序列应连续且无缺失
- 气压数据的合理范围：950-1050 hPa

## 快速开始

### 1. 环境要求
```bash
pip install torch torchvision numpy netCDF4 matplotlib scipy
```

### 2. 数据准备
确保您的数据文件符合上述格式要求，特别是ERA5表面气压数据包含`Psurf_f_ins`变量。

### 3. 运行示例
```python
# 运行完整测试示例
python example.py
```

### 4. 训练模型
```python
from train import ResidualCorrectionTrainer
from config import get_config

# 获取配置
model_config = get_config('model')
data_config = get_config('data')
train_config = get_config('train')

# 更新数据路径
data_config.update({
    'era5_precip_path': '/path/to/your/era5_precipitation.nc',
    'era5_pressure_path': '/path/to/your/era5_surface_pressure.nc',
    'observation_path': '/path/to/your/observations.nc'
})

# 创建训练器
trainer = ResidualCorrectionTrainer(
    model_config=model_config,
    data_config=data_config,
    train_config=train_config,
    save_dir='./experiments/with_pressure'
)

# 开始训练
history = trainer.train()
```

### 5. 模型推理
```python
from inference import PrecipCorrectionInference

# 加载训练好的模型
inference = PrecipCorrectionInference(
    model_path='./experiments/with_pressure/best_model.pth'
)

# 执行校正
stats = inference.correct_precipitation(
    era5_precip_path='/path/to/era5_precipitation.nc',
    era5_pressure_path='/path/to/era5_surface_pressure.nc',  # 包含气压数据
    output_path='./corrected_precipitation.nc'
)
```

## 模型架构

### 输入特征（17通道）
```
通道 0-3:   ERA5降水时间序列特征
通道 4-7:   ERA5温度特征（多层）
通道 8-11:  ERA5湿度特征（多层）
通道 12-15: ERA5风场特征
通道 16:    ERA5表面气压 (Psurf_f_ins) ⭐ 新增
```

### 网络结构
- **编码器**: 残差块 + 注意力机制
- **特征融合**: 专门处理表面气压特征的融合层
- **解码器**: 转置卷积上采样
- **输出**: 残差校正值

### 模型变体
1. **标准模型**: 完整的ResNet-UNet架构，包含注意力机制
2. **轻量级模型**: 适用于资源受限环境的简化版本

## 配置说明

### 模型配置
```python
MODEL_CONFIG = {
    "input_channels": 17,  # 必须为17（包含表面气压）
    "feature_names": [
        # ... 前16个特征
        "Psurf_f_ins"  # 第17个特征：表面气压
    ],
    "pressure_config": {
        "channel_index": 16,  # 气压在第17个通道（索引16）
        "pressure_range": [950, 1050],  # 预期气压范围
        "feature_enhancement": True
    }
}
```

### 训练配置
```python
TRAIN_CONFIG = {
    "pressure_training": {
        "enable": True,
        "warmup_epochs": 10,  # 前N个epoch专注训练气压特征
        "pressure_loss_weight": 1.5  # 气压相关损失权重
    }
}
```

## API参考

### PrecipDataset
```python
dataset = PrecipDataset(
    era5_precip_path="path/to/precip.nc",
    era5_pressure_path="path/to/pressure.nc",  # 新增参数
    observation_path="path/to/obs.nc",
    patch_size=32,
    normalize=True
)

# 返回17通道输入特征
input_features, target = dataset[0]  # input_features.shape = (17, 32, 32)
```

### 模型创建
```python
from model import create_model

model = create_model(
    model_type="standard",
    input_channels=17,  # 必须为17
    hidden_channels=[64, 128, 256, 512],
    use_attention=True
)
```

## 特征重要性分析

模型会自动分析各特征的重要性，包括新增的表面气压特征：

```python
# 获取特征名称
feature_names = dataset.get_feature_names()
print(feature_names)
# ['era5_precip_t0', ..., 'Psurf_f_ins']

# 模型信息
model_info = model.get_model_info()
print(f"包含气压特征: {model_info['includes_pressure']}")
print(f"气压通道索引: {model_info['pressure_channel_index']}")
```

## 评估与验证

### 消融实验
项目支持对比实验来验证表面气压特征的效果：

1. **基线模型**: 16通道（不含气压）
2. **完整模型**: 17通道（含气压）

### 评估指标
- RMSE、MAE、相关系数、偏差
- 空间和时间分析
- 极端事件校正效果
- 特征重要性分析

## 结果输出

### 校正结果文件
```python
# 输出文件包含多个变量
with netCDF4.Dataset('corrected_precipitation.nc', 'r') as f:
    era5_original = f['era5_precipitation'][:]      # 原始ERA5
    corrected = f['corrected_precipitation'][:]     # 校正后结果
    residual = f['residual_field'][:]              # 预测残差
```

### 统计信息
```json
{
  "correction_analysis": {
    "mean_correction_magnitude": 2.15,
    "positive_corrections_ratio": 0.52,
    "negative_corrections_ratio": 0.48,
    "significant_corrections_ratio": 0.35
  }
}
```

## 最佳实践

### 1. 数据质量控制
- 确保气压数据质量和时空一致性
- 检查异常值和缺失值
- 验证数据单位和坐标系

### 2. 模型训练
- 使用渐进式训练策略
- 前期重点训练气压特征
- 监控特征权重变化

### 3. 结果验证
- 与观测数据对比验证
- 分析气压特征的贡献
- 检查空间和时间一致性

## 故障排除

### 常见问题

1. **输入通道数错误**
```
AssertionError: 输入通道数应为17，实际为16
```
解决方案: 确保模型配置中`input_channels=17`

2. **气压数据缺失**
```
KeyError: 'Psurf_f_ins'
```
解决方案: 检查气压数据文件是否包含正确的变量名

3. **维度不匹配**
```
AssertionError: ERA5降水、气压和观测数据维度必须一致
```
解决方案: 确保所有数据具有相同的时空维度

### 调试建议
- 使用`example.py`进行完整系统测试
- 检查`config.py`中的参数设置
- 验证数据文件格式和变量名

## 扩展功能

### 自定义特征
可以轻松添加更多气象特征：

```python
# 在dataset.py中扩展特征
def _compute_additional_features(self):
    # 添加更多特征，如相对湿度、位势高度等
    pass
```

### 模型改进
- 添加时间注意力机制
- 实现多尺度融合
- 集成不确定性估计

## 许可证

本项目基于MIT许可证开源。

## 贡献指南

欢迎提交Issue和Pull Request！在贡献代码前，请确保：
- 遵循现有代码风格
- 添加适当的测试
- 更新相关文档

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**重要提醒**: 本实现将输入特征从16个扩展到17个，新增的ERA5表面气压特征能够提供重要的大气状态信息，有助于改善降水校正效果。所有原有功能保持不变，确保向后兼容性。