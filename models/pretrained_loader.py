"""
预训练权重加载器
将torchvision官方预训练权重注入手写模型
"""
import torch
import torchvision.models as models


def load_pretrained_weights(model, model_name, verbose=True):
    """
    加载ImageNet预训练权重到手写模型
    
    核心逻辑：
    1. 加载torchvision官方模型的state_dict
    2. 遍历手写模型的每个参数key
    3. 形状校验：只加载形状完全匹配的参数
    4. 分类头（fc/classifier最后一层）因形状不一致自动跳过
    
    Args:
        model: 手写模型实例
        model_name: 模型名称，支持 'resnet18', 'vgg16', 'alexnet'
        verbose: 是否打印加载信息
    
    Returns:
        加载了预训练权重的模型
    """
    # 1. 获取官方预训练权重
    if model_name == 'resnet18':
        official_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg16':
        official_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'alexnet':
        official_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    else:
        if verbose:
            print(f"⚠️  模型 {model_name} 不支持预训练权重加载")
        return model
    
    # 2. 提取state_dict
    official_sd = official_model.state_dict()
    our_sd = model.state_dict()
    
    # 3. 形状校验，构建可加载的权重字典
    loadable = {}
    skipped = []
    
    for key in our_sd:
        if key in official_sd:
            # 形状必须完全匹配
            if our_sd[key].shape == official_sd[key].shape:
                loadable[key] = official_sd[key]
            else:
                skipped.append((key, our_sd[key].shape, official_sd[key].shape))
        else:
            # 手写模型中有但官方模型中没有的参数（一般不会发生）
            pass
    
    # 4. 注入权重
    our_sd.update(loadable)
    model.load_state_dict(our_sd)
    
    # 5. 打印加载信息
    if verbose:
        print(f"✅ 成功加载预训练权重: {model_name}")
        print(f"   加载参数: {len(loadable)}/{len(our_sd)}")
        if skipped:
            print(f"   跳过参数（形状不匹配，通常是分类头）:")
            for key, our_shape, off_shape in skipped:
                print(f"     - {key}: our {our_shape} vs official {off_shape}")
    
    return model


def freeze_backbone(model, model_name, strategy='A'):
    """
    冻结模型主干网络的参数
    
    策略:
    - 'A': 冻结全部主干，只训练分类头（适合数据量极少）
    - 'B': 冻结浅层特征，解冻深层特征（平衡效率与性能）
    - 'C': 全参数解冻（适合数据量充足）
    
    Args:
        model: 模型实例
        model_name: 模型名称
        strategy: 冻结策略 'A', 'B', 'C'
    
    Returns:
        应用冻结策略后的模型
    """
    if strategy == 'C':
        # 全参数解冻，不做任何操作
        print("策略 C: 全参数解冻训练")
        return model
    
    if model_name == 'resnet18':
        if strategy == 'A':
            # 冻结除fc外的所有层
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
            print("策略 A: 冻结全部主干，仅训练分类头")
        
        elif strategy == 'B':
            # 冻结conv1, bn1, layer1, layer2；解冻layer3, layer4, fc
            frozen_layers = ['conv1', 'bn1', 'layer1', 'layer2']
            for name, param in model.named_parameters():
                # 检查是否属于冻结层
                should_freeze = any(layer in name for layer in frozen_layers)
                param.requires_grad = not should_freeze
            print("策略 B: 冻结浅层（conv1, bn1, layer1, layer2），解冻深层（layer3, layer4, fc）")
    
    elif model_name in ['vgg16', 'alexnet']:
        if strategy == 'A':
            # 冻结features，只训练classifier
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
            print("策略 A: 冻结全部features，仅训练classifier")
        
        elif strategy == 'B':
            # VGG/AlexNet: 冻结前半部分features，解冻后半部分 + classifier
            # VGG16有31层features，冻结前20层
            # AlexNet有13层features，冻结前8层
            if model_name == 'vgg16':
                freeze_until = 20
            else:  # alexnet
                freeze_until = 8
            
            for idx, (name, param) in enumerate(model.features.named_parameters()):
                param.requires_grad = idx >= freeze_until
            
            # classifier全部解冻
            for param in model.classifier.parameters():
                param.requires_grad = True
            
            print(f"策略 B: 冻结前 {freeze_until} 层features，解冻后续层及classifier")
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def apply_finetune_strategy(model, model_name, strategy, verbose=True):
    """
    应用微调策略的便捷函数
    
    Args:
        model: 模型实例
        model_name: 模型名称
        strategy: 微调策略 'A', 'B', 'C'
        verbose: 是否打印信息
    
    Returns:
        应用了微调策略的模型
    """
    return freeze_backbone(model, model_name, strategy)


if __name__ == '__main__':
    # 测试预训练权重加载
    from resnet import ResNet18
    from vgg import VGG16
    from alexnet import AlexNet
    
    print("=" * 60)
    print("测试 ResNet18 预训练权重加载")
    print("=" * 60)
    model = ResNet18(num_classes=10)
    model = load_pretrained_weights(model, 'resnet18')
    
    print("\n" + "=" * 60)
    print("测试 VGG16 预训练权重加载")
    print("=" * 60)
    model = VGG16(num_classes=10)
    model = load_pretrained_weights(model, 'vgg16')
    
    print("\n" + "=" * 60)
    print("测试 AlexNet 预训练权重加载")
    print("=" * 60)
    model = AlexNet(num_classes=10)
    model = load_pretrained_weights(model, 'alexnet')
    
    print("\n" + "=" * 60)
    print("测试冻结策略 B")
    print("=" * 60)
    model = ResNet18(num_classes=10)
    model = load_pretrained_weights(model, 'resnet18', verbose=False)
    model = freeze_backbone(model, 'resnet18', strategy='B')
