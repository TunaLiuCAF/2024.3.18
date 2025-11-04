"""
降水残差校正模型
支持17个输入通道 (包含ERA5表面气压 Psurf_f_ins)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class AttentionModule(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(AttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class PrecipResidualCorrectionModel(nn.Module):
    """
    降水残差校正模型
    
    输入：17通道特征 (原16个特征 + ERA5表面气压 Psurf_f_ins)
    输出：残差校正值
    """
    
    def __init__(
        self,
        input_channels: int = 17,  # 修改为17，包含表面气压
        hidden_channels: list = [64, 128, 256, 512],
        output_channels: int = 1,
        use_attention: bool = True,
        dropout: float = 0.1
    ):
        """
        初始化模型
        
        Args:
            input_channels: 输入通道数 (17，包含Psurf_f_ins)
            hidden_channels: 隐藏层通道数列表
            output_channels: 输出通道数
            use_attention: 是否使用注意力机制
            dropout: dropout率
        """
        super(PrecipResidualCorrectionModel, self).__init__()
        
        self.input_channels = input_channels
        self.use_attention = use_attention
        
        # 输入特征处理层
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels[0], kernel_size=7, 
                     padding=3, bias=False),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 编码器 - 残差块层
        self.encoder_layers = nn.ModuleList()
        in_ch = hidden_channels[0]
        
        for i, out_ch in enumerate(hidden_channels[1:]):
            # 每层包含多个残差块
            layer_blocks = nn.Sequential(
                ResidualBlock(in_ch, out_ch, stride=2),
                ResidualBlock(out_ch, out_ch),
                ResidualBlock(out_ch, out_ch)
            )
            self.encoder_layers.append(layer_blocks)
            
            # 添加注意力模块
            if self.use_attention:
                attention = AttentionModule(out_ch)
                self.encoder_layers.append(attention)
            
            in_ch = out_ch
        
        # 特征融合层 - 专门处理表面气压特征
        self.pressure_fusion = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-1] // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[-1] // 2, hidden_channels[-1], kernel_size=1),
            nn.BatchNorm2d(hidden_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        # 解码器 - 上采样层
        self.decoder_layers = nn.ModuleList()
        
        for i in range(len(hidden_channels) - 1, 0, -1):
            decoder_block = nn.Sequential(
                nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i-1], 
                                 kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_channels[i-1]),
                nn.ReLU(inplace=True),
                ResidualBlock(hidden_channels[i-1], hidden_channels[i-1])
            )
            self.decoder_layers.append(decoder_block)
        
        # 最终上采样到原始尺寸
        self.final_upsample = nn.ConvTranspose2d(
            hidden_channels[0], hidden_channels[0] // 2,
            kernel_size=4, stride=2, padding=1
        )
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[0] // 2, hidden_channels[0] // 4, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[0] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels[0] // 4, output_channels, 
                     kernel_size=1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征张量 (batch_size, 17, height, width)
               包含16个原始特征 + 1个表面气压特征
        
        Returns:
            residual: 残差校正值 (batch_size, 1, height, width)
        """
        # 验证输入维度
        assert x.size(1) == self.input_channels, \
            f"输入通道数应为{self.input_channels}，实际为{x.size(1)}"
        
        # 保存原始尺寸用于最终输出
        original_size = x.shape[-2:]
        
        # 输入特征处理
        x = self.input_conv(x)
        
        # 编码器阶段
        encoder_features = []
        for layer in self.encoder_layers:
            x = layer(x)
            if isinstance(layer, nn.Sequential):  # 残差块层
                encoder_features.append(x)
        
        # 特征融合 - 强化表面气压信息
        x = self.pressure_fusion(x)
        
        # 解码器阶段
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)
            
            # 跳跃连接 (可选)
            if i < len(encoder_features) - 1:
                encoder_feat = encoder_features[-(i+2)]
                if x.shape[-2:] == encoder_feat.shape[-2:]:
                    x = x + encoder_feat
        
        # 最终上采样
        x = self.final_upsample(x)
        
        # 确保输出尺寸与输入一致
        if x.shape[-2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', 
                            align_corners=False)
        
        # 输出层
        residual = self.output_conv(x)
        
        return residual
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'PrecipResidualCorrectionModel',
            'input_channels': self.input_channels,
            'includes_pressure': True,
            'pressure_channel_index': 16,  # 表面气压在第17个通道 (索引16)
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'use_attention': self.use_attention
        }


class LightweightResidualModel(nn.Module):
    """
    轻量级残差校正模型
    适用于资源受限环境
    """
    
    def __init__(
        self,
        input_channels: int = 17,
        hidden_channels: int = 64,
        output_channels: int = 1
    ):
        super(LightweightResidualModel, self).__init__()
        
        self.input_channels = input_channels
        
        # 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(
            input_channels, input_channels, kernel_size=3, 
            padding=1, groups=input_channels
        )
        self.pointwise_conv = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=1
        )
        
        # 特征处理
        self.feature_conv = nn.Sequential(
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.output_conv = nn.Conv2d(
            hidden_channels // 2, output_channels, kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        assert x.size(1) == self.input_channels, \
            f"输入通道数应为{self.input_channels}，实际为{x.size(1)}"
        
        # 深度可分离卷积
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        
        # 特征处理
        x = self.feature_conv(x)
        
        # 输出
        residual = self.output_conv(x)
        
        return residual


def create_model(model_type: str = "standard", **kwargs) -> nn.Module:
    """
    模型工厂函数
    
    Args:
        model_type: 模型类型 ('standard' 或 'lightweight')
        **kwargs: 模型参数
    
    Returns:
        模型实例
    """
    if model_type == "standard":
        return PrecipResidualCorrectionModel(**kwargs)
    elif model_type == "lightweight":
        return LightweightResidualModel(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


if __name__ == "__main__":
    # 测试模型
    print("测试降水残差校正模型...")
    
    # 创建模型
    model = create_model("standard", input_channels=17)
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试前向传播
    batch_size = 2
    height, width = 32, 32
    
    # 创建17通道输入 (包含表面气压)
    x = torch.randn(batch_size, 17, height, width)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape}")
    print("模型测试成功！")
    
    # 测试轻量级模型
    print("\n测试轻量级模型...")
    lightweight_model = create_model("lightweight", input_channels=17)
    
    with torch.no_grad():
        output_light = lightweight_model(x)
    
    print(f"轻量级模型输出shape: {output_light.shape}")
    print("轻量级模型测试成功！")