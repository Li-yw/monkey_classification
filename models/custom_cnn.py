"""
SimpleCNN - 基础卷积神经网络
教学用途：展示CNN的基本组成
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    简单的5层卷积神经网络
    结构：Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),  # 使用 inplace=False 以兼容 Grad-CAM
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分类器部分
        # 输入图像 224x224，经过5次下采样后为 7x7
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def simple_cnn(num_classes=10, **kwargs):
    """构造SimpleCNN模型的便捷函数"""
    return SimpleCNN(num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    # 测试模型
    model = SimpleCNN(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
