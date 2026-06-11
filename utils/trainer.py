"""
模型训练与评估工具
职责：
- 训练循环实现
- 验证评估
- 学习率调度
"""
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_EPOCHS, LEARNING_RATE, PRETRAINED_LR, LR_STEP_SIZE, LR_GAMMA, DEVICE, CHECKPOINT_DIR


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    """
    训练一个epoch
    
    完整呈现梯度清零、前向传播、损失计算、反向传播、参数更新的五步流程
    
    Args:
        model: 模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        num_epochs: 总epoch数
    
    Returns:
        epoch_loss: 平均损失
        epoch_acc: 准确率
    """
    model.train()  # 切换到训练模式
    
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # ========== 五步标准流程 ==========
        
        # 步骤1: 梯度清零
        optimizer.zero_grad()
        
        # 步骤2: 前向传播
        outputs = model(inputs)
        
        # 步骤3: 计算损失
        loss = criterion(outputs, labels)
        
        # 步骤4: 反向传播（计算梯度）
        loss.backward()
        
        # 步骤5: 参数更新
        optimizer.step()
        
        # ========== 统计信息 ==========
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/total_samples:.4f}',
            'acc': f'{running_corrects/total_samples:.4f}'
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, class_names=None):
    """
    模型评估
    
    Args:
        model: 模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        class_names: 类别名称列表
    
    Returns:
        eval_loss: 平均损失
        eval_acc: 准确率
        f1: 加权F1分数
        report: 分类报告（字典）
        cm: 混淆矩阵
        all_preds: 所有预测结果
        all_labels: 所有真实标签
    """
    model.eval()  # 切换到评估模式
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 预测
            _, preds = torch.max(outputs, 1)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    eval_loss = running_loss / len(all_labels)
    eval_acc = np.mean(all_preds == all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 分类报告
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(all_labels)))]
    report = classification_report(all_labels, all_preds, target_names=class_names, 
                                    output_dict=True, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return eval_loss, eval_acc, f1, report, cm, all_preds, all_labels


def train_model(model, train_loader, val_loader, num_epochs=None, 
                learning_rate=None, device=None, use_pretrained=False,
                save_best=True, checkpoint_dir=None, verbose=True):
    """
    完整的训练流程
    
    包含：
    - 训练循环
    - 验证评估
    - 学习率调度
    - 最佳模型保存
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        use_pretrained: 是否使用预训练模型
        save_best: 是否保存最佳模型
        checkpoint_dir: 检查点保存目录
        verbose: 是否打印详细信息
    
    Returns:
        model: 训练后的模型
        history: 训练历史记录
    """
    # 参数默认值
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    if learning_rate is None:
        learning_rate = PRETRAINED_LR if use_pretrained else LEARNING_RATE
    if device is None:
        device = DEVICE
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }
    
    # 最佳模型跟踪
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # 训练开始时间
    start_time = time.time()
    
    if verbose:
        print("=" * 80)
        print(f"开始训练 - 共 {num_epochs} 轮")
        print(f"设备: {device}")
        print(f"学习率: {learning_rate}")
        print(f"优化器: Adam")
        print(f"学习率调度: 每 {LR_STEP_SIZE} 轮衰减 {LR_GAMMA}")
        print("=" * 80)
    
    # 训练循环
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
        
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # 验证
        val_loss, val_acc, val_f1, _, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # 打印结果
        if verbose:
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            
            if save_best:
                best_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, best_path)
                if verbose:
                    print(f"✅ 保存最佳模型 (Acc: {best_acc:.4f})")
    
    # 训练结束
    time_elapsed = time.time() - start_time
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"训练完成！")
        print(f"总耗时: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒")
        print(f"最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch})")
        print("=" * 80)
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    return model, history


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """
    保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 指标字典
        filepath: 保存路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, filepath)


def load_checkpoint(model, optimizer, filepath):
    """
    加载训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        filepath: 检查点路径
    
    Returns:
        model: 加载权重后的模型
        optimizer: 加载状态后的优化器
        epoch: epoch数
        metrics: 指标字典
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    return model, optimizer, epoch, metrics


if __name__ == '__main__':
    print("训练工具模块")
    print("主要函数:")
    print("  - train_one_epoch: 训练一个epoch")
    print("  - evaluate: 模型评估")
    print("  - train_model: 完整训练流程")
