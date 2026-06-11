"""
可视化工具
职责：
- 训练历史可视化
- 混淆矩阵可视化
- Grad-CAM可视化
- 样本展示
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MEAN, STD


# ========== 训练历史可视化 ==========

def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典，包含train_loss, train_acc, val_loss, val_acc, val_f1, lr
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('训练与验证损失', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('训练与验证准确率', fontsize=14)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1-Score曲线
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1-Score', fontsize=12)
    axes[1, 0].set_title('验证集F1分数', fontsize=14)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1, 1].plot(epochs, history['lr'], 'm-', label='Learning Rate', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('学习率变化', fontsize=14)
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练历史曲线已保存至: {save_path}")
    
    plt.show()


# ========== 混淆矩阵可视化 ==========

def plot_confusion_matrix(cm, class_names, save_path=None, normalize=False):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        normalize: 是否归一化
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = '归一化混淆矩阵'
        fmt = '.2f'
    else:
        title = '混淆矩阵（原始计数）'
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 使用seaborn绘制热力图
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={'label': '样本数' if not normalize else '比例'}
    )
    
    ax.set_xlabel('预测类别', fontsize=12)
    ax.set_ylabel('真实类别', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    
    plt.show()


# ========== 样本可视化 ==========

def plot_sample_images(dataloader, class_names, num_samples=8, save_path=None):
    """
    展示样本图像
    
    Args:
        dataloader: 数据加载器
        class_names: 类别名称列表
        num_samples: 展示样本数
        save_path: 保存路径
    """
    # 获取一个batch
    images, labels = next(iter(dataloader))
    
    # 限制显示数量
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(3 * num_samples // 2, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].clone()
        
        # 反标准化
        for t, m, s in zip(img, MEAN, STD):
            t.mul_(s).add_(m)
        
        # 转为numpy并调整通道顺序
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].set_title(class_names[labels[i]], fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"样本图像已保存至: {save_path}")
    
    plt.show()


def plot_class_distribution(class_counts, class_names, save_path=None):
    """
    绘制类别分布图
    
    Args:
        class_counts: 各类别样本数
        class_names: 类别名称列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_counts)))
    
    # 柱状图
    axes[0].bar(range(len(class_counts)), class_counts, color=colors)
    axes[0].set_xlabel('类别', fontsize=12)
    axes[0].set_ylabel('样本数', fontsize=12)
    axes[0].set_title('类别分布（柱状图）', fontsize=14)
    axes[0].set_xticks(range(len(class_counts)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    
    for i, count in enumerate(class_counts):
        axes[0].text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10)
    
    # 饼图
    axes[1].pie(class_counts, labels=class_names, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    axes[1].set_title('类别分布（饼图）', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"类别分布图已保存至: {save_path}")
    
    plt.show()


# ========== Grad-CAM 实现 ==========

class GradCAM:
    """
    梯度加权类激活映射（Grad-CAM）
    
    手动实现，无需第三方库
    
    原理：
    1. 前向传播，捕获目标层特征图（activation）
    2. 反向传播，捕获目标层梯度（gradient）
    3. 对梯度进行全局平均池化得到权重
    4. 加权求和特征图，ReLU激活
    5. 上采样到输入尺寸
    """
    
    def __init__(self, model, target_layer):
        """
        初始化Grad-CAM
        
        Args:
            model: 模型
            target_layer: 目标卷积层（用于可视化）
        """
        self.model = model
        self.target_layer = target_layer
        
        # 存储特征图和梯度
        self.activations = None
        self.gradients = None
        
        # 注册hook
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """保存前向传播的特征图"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_tensor, class_idx=None):
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入张量 (1, 3, H, W)
            class_idx: 目标类别索引，None则使用预测类别
        
        Returns:
            cam: 热力图，范围[0, 1]，大小与输入一致
        """
        # 确保模型在评估模式
        self.model.eval()
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 如果未指定类别，使用预测类别
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # 获取特征图和梯度
        activations = self.activations[0]  # (C, H', W')
        gradients = self.gradients[0]      # (C, H', W')
        
        # 计算权重：全局平均池化
        weights = gradients.mean(dim=[1, 2])  # (C,)
        
        # 加权求和
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)  # (H', W')
        
        # ReLU激活
        cam = torch.relu(cam)
        
        # 归一化到[0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # 转为numpy
        cam = cam.cpu().numpy()
        
        # 上采样到输入尺寸
        input_size = input_tensor.shape[2:]  # (H, W)
        cam = cv2.resize(cam, input_size[::-1])
        
        return cam
    
    def remove_hooks(self):
        """清理hook，释放内存"""
        self.forward_hook.remove()
        self.backward_hook.remove()


def visualize_gradcam(model, image_tensor, class_idx, class_names, 
                      target_layer=None, save_path=None):
    """
    可视化Grad-CAM
    
    Args:
        model: 模型
        image_tensor: 输入图像张量 (1, 3, H, W)
        class_idx: 真实类别索引
        class_names: 类别名称列表
        target_layer: 目标层（None则自动选择）
        save_path: 保存路径
    """
    # 自动选择目标层
    if target_layer is None:
        target_layer = auto_select_target_layer(model)
    
    # 创建Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 生成热力图
    cam = grad_cam(image_tensor)
    
    # 获取预测结果
    with torch.no_grad():
        output = model(image_tensor)
        pred_idx = output.argmax(dim=1).item()
    
    # 准备原图
    img = image_tensor[0].clone().cpu()
    for t, m, s in zip(img, MEAN, STD):
        t.mul_(s).add_(m)
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # 创建热力图叠加
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * img_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('原始图像', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM热力图', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'叠加图\n真实: {class_names[class_idx]}\n预测: {class_names[pred_idx]}', 
                      fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 清理hook
    grad_cam.remove_hooks()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM可视化已保存至: {save_path}")
    
    plt.show()


def auto_select_target_layer(model):
    """
    自动选择目标层
    
    不同模型的目标层选择策略：
    - ResNet: layer4最后一个残差块
    - VGG/AlexNet: features最后一个卷积层（不是ReLU）
    - SimpleCNN: features最后一个卷积层
    """
    model_name = model.__class__.__name__
    
    if 'ResNet' in model_name:
        # ResNet: layer4的最后一个残差块
        return model.layer4[-1]
    
    elif 'VGG' in model_name or 'AlexNet' in model_name:
        # VGG/AlexNet: features中倒数第3个层（最后一个Conv）
        # features结构: [..., Conv, ReLU, MaxPool]
        return model.features[-3]
    
    elif 'SimpleCNN' in model_name:
        # SimpleCNN: 找到最后一个Conv层
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                return layer
    
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")


def plot_gradcam_comparison(model, dataloader, class_names, num_samples=4, 
                            target_layer=None, save_path=None):
    """
    批量可视化Grad-CAM对比
    
    Args:
        model: 模型
        dataloader: 数据加载器
        class_names: 类别名称列表
        num_samples: 样本数
        target_layer: 目标层
        save_path: 保存路径
    """
    model.eval()
    
    # 自动选择目标层
    if target_layer is None:
        target_layer = auto_select_target_layer(model)
    
    # 获取样本
    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # 创建Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        img_tensor = images[i:i+1]
        
        # 生成Grad-CAM
        cam = grad_cam(img_tensor)
        
        # 预测
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = output.argmax(dim=1).item()
        
        # 准备原图
        img = img_tensor[0].clone().cpu()
        for t, m, s in zip(img, MEAN, STD):
            t.mul_(s).add_(m)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # 叠加
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * img_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        # 绘制
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'原图 - {class_names[labels[i]]}', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].set_title('Grad-CAM', fontsize=10)
        axes[i, 1].axis('off')
        
        correct = '✓' if pred_idx == labels[i].item() else '✗'
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'叠加 - 预测: {class_names[pred_idx]} {correct}', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # 清理hook
    grad_cam.remove_hooks()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM对比图已保存至: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    print("可视化工具模块")
    print("主要功能:")
    print("  - plot_training_history: 训练历史曲线")
    print("  - plot_confusion_matrix: 混淆矩阵")
    print("  - plot_sample_images: 样本展示")
    print("  - GradCAM: Grad-CAM可视化类")
    print("  - visualize_gradcam: 单张Grad-CAM可视化")
    print("  - plot_gradcam_comparison: 批量Grad-CAM对比")
