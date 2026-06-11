"""
ResNet18 - 2015年ImageNet冠军模型
教学用途：展示残差连接如何解决深层网络退化问题

⚠️ 重要：命名必须与torchvision.models.resnet18完全一致
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    ResNet基础残差块
    结构：conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> + identity
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # ⚠️ 命名必须与torchvision完全一致
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)  # 使用 inplace=False 以兼容 Grad-CAM
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 如果维度不匹配，需要下采样
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接：F(x) + x
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    ResNet18架构
    输入：224x224 RGB图像
    
    结构特点：
    - 18层网络（含Stem层）
    - 使用BasicBlock（每个块2层卷积）
    - 4个残差阶段：layer1-layer4
    - 残差连接解决深层网络退化问题
    """
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        
        self.in_channels = 64
        
        # ⚠️ 命名必须与torchvision完全一致
        
        # Stem层：7x7卷积 + 最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)  # 使用 inplace=False 以兼容 Grad-CAM
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差阶段
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        构建残差阶段
        
        Args:
            block: BasicBlock类型
            out_channels: 输出通道数
            num_blocks: 该阶段的残差块数量
            stride: 第一个块的步长
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 维度不匹配时，需要用1x1卷积调整
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差阶段
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def resnet18(num_classes=10, **kwargs):
    """构造ResNet18模型的便捷函数"""
    return ResNet18(num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    # 测试模型
    model = ResNet18(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 打印各层名称（用于验证预训练权重加载）
    print("\nLayer names:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            print(f"  {name}")
