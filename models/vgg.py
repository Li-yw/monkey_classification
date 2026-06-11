"""
VGG16 - 2014年ImageNet亚军模型
教学用途：展示小卷积核堆叠的威力

⚠️ 重要：命名必须与torchvision.models.vgg16完全一致
"""
import torch
import torch.nn as nn


class VGG16(nn.Module):
    """
    VGG16架构
    输入：224x224 RGB图像
    
    结构特点：
    - 全部使用3x3小卷积核
    - 13个卷积层 + 3个全连接层
    - 网络深度达到16层
    - 参数量巨大（约138M）
    """
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        
        # ⚠️ 命名必须与torchvision完全一致
        # features中的索引必须精确对应
        self.features = nn.Sequential(
            # Block 1: 64通道
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # 0
            nn.ReLU(inplace=False),                        # 1  # 使用 inplace=False 以兼容 Grad-CAM
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 2
            nn.ReLU(inplace=False),                        # 3
            nn.MaxPool2d(kernel_size=2, stride=2),         # 4
            
            # Block 2: 128通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 5
            nn.ReLU(inplace=False),                        # 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 7
            nn.ReLU(inplace=False),                        # 8
            nn.MaxPool2d(kernel_size=2, stride=2),         # 9
            
            # Block 3: 256通道
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 10
            nn.ReLU(inplace=False),                        # 11
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 12
            nn.ReLU(inplace=False),                        # 13
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 14
            nn.ReLU(inplace=False),                        # 15
            nn.MaxPool2d(kernel_size=2, stride=2),         # 16
            
            # Block 4: 512通道
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # 17
            nn.ReLU(inplace=False),                        # 18
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 19
            nn.ReLU(inplace=False),                        # 20
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 21
            nn.ReLU(inplace=False),                        # 22
            nn.MaxPool2d(kernel_size=2, stride=2),         # 23
            
            # Block 5: 512通道
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 24
            nn.ReLU(inplace=False),                        # 25
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 26
            nn.ReLU(inplace=False),                        # 27
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 28 - 最后一个卷积层
            nn.ReLU(inplace=False),                        # 29
            nn.MaxPool2d(kernel_size=2, stride=2),         # 30
        )
        
        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 分类器
        # ⚠️ index 6 是最后的FC层，输出num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 0
            nn.ReLU(inplace=False),        # 1
            nn.Dropout(),                  # 2
            nn.Linear(4096, 4096),         # 3
            nn.ReLU(inplace=False),        # 4
            nn.Dropout(),                  # 5
            nn.Linear(4096, num_classes),  # 6 - 最后一个全连接层
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg16(num_classes=10, **kwargs):
    """构造VGG16模型的便捷函数"""
    return VGG16(num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    # 测试模型
    model = VGG16(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 确认features的层索引（Grad-CAM需要用到）
    print("\nLast 5 layers in features:")
    for idx in range(26, 31):
        print(f"  {idx}: {model.features[idx]}")
