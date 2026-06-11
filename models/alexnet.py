"""
AlexNet - 2012年ImageNet冠军模型
教学用途：展示深度学习时代的开端

⚠️ 重要：命名必须与torchvision.models.alexnet完全一致
"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet架构
    输入：224x224 RGB图像
    
    结构特点：
    - 5个卷积层（features）
    - 3个全连接层（classifier）
    - 使用ReLU激活函数（首次大规模应用）
    - 使用Dropout防止过拟合
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # ⚠️ 命名必须与torchvision完全一致
        self.features = nn.Sequential(
            # Conv1: 3 -> 64, 11x11, stride=4, padding=2
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),  # 使用 inplace=False 以兼容 Grad-CAM
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 64 -> 192, 5x5, padding=2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 192 -> 384, 3x3, padding=1
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            
            # Conv4: 384 -> 256, 3x3, padding=1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            
            # Conv5: 256 -> 256, 3x3, padding=1
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 自适应平均池化，将特征图调整为固定大小
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 分类器
        # ⚠️ index 6 是最后的FC层，会被替换
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            # index 6: 最后一个全连接层，输出num_classes
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(num_classes=10, **kwargs):
    """构造AlexNet模型的便捷函数"""
    return AlexNet(num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    # 测试模型
    model = AlexNet(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 打印features的层索引（用于确认命名）
    print("\nFeatures layers:")
    for idx, layer in enumerate(model.features):
        print(f"  {idx}: {layer}")
