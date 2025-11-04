"""
降水残差校正模型训练流程
支持ERA5表面气压 (Psurf_f_ins) 作为输入特征
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import logging

from dataset import PrecipDataset
from model import create_model


class ResidualCorrectionTrainer:
    """降水残差校正模型训练器"""
    
    def __init__(
        self,
        model_config: dict,
        data_config: dict,
        train_config: dict,
        save_dir: str = "./checkpoints"
    ):
        """
        初始化训练器
        
        Args:
            model_config: 模型配置
            data_config: 数据配置
            train_config: 训练配置
            save_dir: 模型保存目录
        """
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = create_model(**model_config)
        self.model.to(self.device)
        
        # 打印模型信息
        model_info = self.model.get_model_info()
        print(f"模型信息: {model_info}")
        print(f"模型包含表面气压特征: {model_info['includes_pressure']}")
        
        # 设置损失函数
        self.criterion = self._setup_loss_function()
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._setup_scheduler()
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # 设置日志
        self._setup_logging()
    
    def _setup_loss_function(self):
        """设置损失函数"""
        loss_type = self.train_config.get('loss_type', 'mse')
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()
        elif loss_type == 'combined':
            # 组合损失：MSE + MAE
            class CombinedLoss(nn.Module):
                def __init__(self, mse_weight=0.7, mae_weight=0.3):
                    super().__init__()
                    self.mse = nn.MSELoss()
                    self.mae = nn.L1Loss()
                    self.mse_weight = mse_weight
                    self.mae_weight = mae_weight
                
                def forward(self, pred, target):
                    return (self.mse_weight * self.mse(pred, target) + 
                           self.mae_weight * self.mae(pred, target))
            
            return CombinedLoss()
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    def _setup_optimizer(self):
        """设置优化器"""
        optimizer_type = self.train_config.get('optimizer', 'adam')
        lr = self.train_config.get('learning_rate', 1e-3)
        weight_decay = self.train_config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self.train_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, 
                           momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        scheduler_type = self.train_config.get('scheduler', 'step')
        
        if scheduler_type == 'step':
            step_size = self.train_config.get('step_size', 10)
            gamma = self.train_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = self.train_config.get('T_max', 50)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'reduce':
            factor = self.train_config.get('factor', 0.5)
            patience = self.train_config.get('patience', 5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience
            )
        else:
            return None
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练和验证数据
        
        Returns:
            train_loader, val_loader
        """
        self.logger.info("正在准备数据集...")
        
        # 创建完整数据集
        full_dataset = PrecipDataset(**self.data_config)
        
        # 分割训练和验证集
        train_ratio = self.train_config.get('train_ratio', 0.8)
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据加载器
        batch_size = self.train_config.get('batch_size', 16)
        num_workers = self.train_config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        self.logger.info(f"训练样本数: {len(train_dataset)}")
        self.logger.info(f"验证样本数: {len(val_dataset)}")
        self.logger.info(f"特征维度: {full_dataset.get_feature_names()}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 移到设备
            inputs = inputs.to(self.device)  # (batch_size, 17, H, W)
            targets = targets.to(self.device)  # (batch_size, H, W)
            
            # 确保目标维度正确
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)  # (batch_size, 1, H, W)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            grad_clip = self.train_config.get('grad_clip', None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}'
                )
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self) -> Dict:
        """
        主训练循环
        
        Returns:
            训练历史字典
        """
        # 准备数据
        train_loader, val_loader = self.prepare_data()
        
        # 训练参数
        num_epochs = self.train_config.get('num_epochs', 100)
        save_interval = self.train_config.get('save_interval', 10)
        early_stopping_patience = self.train_config.get('early_stopping_patience', 20)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"开始训练，共 {num_epochs} 个epoch")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate_epoch(val_loader)
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # 计算epoch时间
            epoch_time = time.time() - start_time
            
            # 打印信息
            self.logger.info(
                f'Epoch {epoch+1}/{num_epochs} - '
                f'Train Loss: {train_loss:.6f}, '
                f'Val Loss: {val_loss:.6f}, '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.8f}, '
                f'Time: {epoch_time:.2f}s'
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, 'best_model.pth', val_loss)
                self.logger.info(f'保存最佳模型，验证损失: {val_loss:.6f}')
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth', val_loss)
            
            # 早停
            if patience_counter >= early_stopping_patience:
                self.logger.info(f'早停触发，已等待 {patience_counter} 个epoch')
                break
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        self.logger.info("训练完成！")
        
        return self.train_history
    
    def save_checkpoint(self, epoch: int, filename: str, val_loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'model_config': self.model_config,
            'train_history': self.train_history
        }
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def save_training_history(self):
        """保存训练历史"""
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        ax1.plot(epochs, self.train_history['train_loss'], label='Train Loss')
        ax1.plot(epochs, self.train_history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 学习率曲线
        ax2.plot(epochs, self.train_history['learning_rate'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数 - 训练示例"""
    
    # 模型配置
    model_config = {
        'model_type': 'standard',
        'input_channels': 17,  # 包含表面气压的17通道
        'hidden_channels': [64, 128, 256, 512],
        'output_channels': 1,
        'use_attention': True,
        'dropout': 0.1
    }
    
    # 数据配置
    data_config = {
        'era5_precip_path': '/path/to/era5_precipitation.nc',
        'era5_pressure_path': '/path/to/era5_surface_pressure.nc',  # 包含Psurf_f_ins
        'observation_path': '/path/to/observations.nc',
        'patch_size': 32,
        'date_start': '2008-01-01',
        'date_end': '2019-12-31',
        'normalize': True
    }
    
    # 训练配置
    train_config = {
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'loss_type': 'combined',
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'train_ratio': 0.8,
        'num_workers': 4,
        'save_interval': 10,
        'early_stopping_patience': 20
    }
    
    # 创建训练器
    trainer = ResidualCorrectionTrainer(
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
        save_dir='./experiments/exp_with_pressure'
    )
    
    # 开始训练
    history = trainer.train()
    
    print("训练完成！")
    print(f"最佳验证损失: {min(history['val_loss']):.6f}")


if __name__ == "__main__":
    main()