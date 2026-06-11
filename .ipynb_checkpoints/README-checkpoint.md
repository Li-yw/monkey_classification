# 🐒 10 Monkey Species - 深度学习图像分类实战项目

面向深度学习初学者的图像分类实战教学项目，以 Kaggle 公开数据集「10 Monkey Species」为实验载体，系统性地覆盖了从数据处理到模型部署的完整深度学习工程流程。

## 📋 项目特点

- **🎓 教学导向**：每一个步骤都有详细的原理解释，代码可读性强
- **🔨 手写模型**：从 `nn.Module` 出发手写实现 SimpleCNN、AlexNet、VGG16、ResNet18
- **🚀 迁移学习**：完整的预训练权重加载与微调策略实现
- **🔍 可解释性**：手写 Grad-CAM 可视化，理解模型决策过程
- **📊 对比实验**：四个模型在准确率、F1-Score、训练时长、参数量维度的横向对比

## 📁 项目结构

```
monkey_classification/
├── 📓 01_single_model.ipynb      # 单模型端到端流程（详细原理）
├── 📓 02_comparison.ipynb        # 四模型对比实验
├── models/
│   ├── __init__.py
│   ├── custom_cnn.py             # SimpleCNN 基础卷积网络
│   ├── alexnet.py                # AlexNet（2012 ImageNet冠军）
│   ├── vgg.py                    # VGG16（2014 ImageNet亚军）
│   ├── resnet.py                 # ResNet18（2015 ImageNet冠军）
│   └── pretrained_loader.py      # 预训练权重加载器
├── utils/
│   ├── __init__.py
│   ├── dataset.py                # 数据加载与增强
│   ├── trainer.py                # 训练与评估
│   └── visualize.py              # 可视化工具 + Grad-CAM
├── config.py                     # 全局配置
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明
```

## 🗂️ 数据集

**10 Monkey Species** 数据集来自 Kaggle，包含约 1400 张灵长类动物图像，涵盖 10 个物种：

| 编号 | 类别名 | 中文名 |
|------|--------|--------|
| n0 | mantled_howler | 赤吼猴 |
| n1 | patas_monkey | 赤猴 |
| n2 | bald_uakari | 秃猴 |
| n3 | japanese_macaque | 日本猕猴 |
| n4 | pygmy_marmoset | 侏狨 |
| n5 | white_headed_capuchin | 白头卷尾猴 |
| n6 | silvery_marmoset | 银狨 |
| n7 | common_squirrel_monkey | 松鼠猴 |
| n8 | black_headed_night_monkey | 黑头夜猴 |
| n9 | nilgiri_langur | 尼尔吉里叶猴 |

**数据集下载：** [Kaggle - 10 Monkey Species](https://www.kaggle.com/datasets/slothkong/10-monkey-species)

下载后解压到 `data/10_monkey_species/` 目录，或修改 `config.py` 中的 `DATA_DIR`。

## 🚀 快速开始

### 1. 安装依赖

```bash
cd monkey_classification
pip install -r requirements.txt
```

### 2. 下载数据集

从 Kaggle 下载数据集并解压：

```bash
# 创建数据目录
mkdir -p data/10_monkey_species

# 解压数据集到该目录
# 确保目录结构为：
# data/10_monkey_species/
# ├── training/training/
# │   ├── n0/
# │   ├── n1/
# │   └── ...
# └── validation/validation/
#     ├── n0/
#     ├── n1/
#     └── ...
```

### 3. 运行 Notebook

```bash
jupyter notebook
```

打开 `01_single_model.ipynb` 开始学习单模型完整流程，或打开 `02_comparison.ipynb` 查看四模型对比实验。

## ⚙️ 配置说明

主要配置项在 `config.py` 中：

```python
# 数据集路径
DATA_DIR = "./data/10_monkey_species"

# 训练超参数
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3          # 从零训练
PRETRAINED_LR = 1e-4          # 预训练微调

# 迁移学习
USE_PRETRAINED = True         # 是否使用预训练权重
FINETUNE_STRATEGY = 'B'       # 微调策略：'A', 'B', 'C'
```

### 微调策略说明

| 策略 | 冻结方式 | 适用场景 |
|------|----------|----------|
| A | 冻结全部主干，仅训练分类头 | 数据量极少（<100样本/类）|
| B | 冻结浅层，解冻深层 | 平衡效率与性能（推荐）|
| C | 全参数解冻，小学习率微调 | 数据量充足（>500样本/类）|

## 📊 实验结果示例

基于 ImageNet 预训练权重，策略 B 微调：

| 模型 | 参数量 | 准确率 | F1-Score | 训练时长 |
|------|--------|--------|----------|----------|
| SimpleCNN | 26.8M | ~0.85 | ~0.84 | 较短 |
| AlexNet | 57.0M | ~0.92 | ~0.91 | 中等 |
| VGG16 | 134.3M | ~0.94 | ~0.93 | 较长 |
| ResNet18 | 11.2M | ~0.96 | ~0.95 | 中等 |

*注：实际结果可能因硬件和数据集版本而异*

## 🎓 学习路径

### 初学者路径
1. 运行 `01_single_model.ipynb`，理解每个步骤
2. 阅读 `models/` 下的模型实现，理解CNN架构
3. 修改 `config.py` 中的超参数，观察训练效果
4. 尝试不同的微调策略（A/B/C）

### 进阶路径
1. 运行 `02_comparison.ipynb`，对比四个模型
2. 分析 Grad-CAM 可视化，理解模型决策
3. 尝试添加新的模型（如 EfficientNet、MobileNet）
4. 实验不同的数据增强策略

## 🔧 常见问题

### Q1: num_workers > 0 在 Windows 报错？
**A:** Windows 下多进程数据加载有问题，设置 `NUM_WORKERS = 0`。

### Q2: 验证集准确率很低？
**A:** 检查是否忘记 `model.eval()` 和 `torch.no_grad()`。

### Q3: 预训练模型训练时 loss 爆炸？
**A:** 预训练模型使用更小的学习率（1e-4），避免破坏预训练权重。

### Q4: Grad-CAM 运行时内存不足？
**A:** 用完后调用 `grad_cam.remove_hooks()` 清理 hook。

### Q5: 加载预训练权重报 key 不匹配？
**A:** 模型命名必须与 torchvision 完全一致，或使用 `strict=False`。

## 📚 参考资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [torchvision 模型](https://pytorch.org/vision/stable/models.html)
- [Grad-CAM 论文](https://arxiv.org/abs/1610.02391)
- [ResNet 论文](https://arxiv.org/abs/1512.03385)
- [VGG 论文](https://arxiv.org/abs/1409.1556)
- [AlexNet 论文](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

## 📄 许可证

本项目仅供学习和教学使用。数据集版权归 Kaggle 及原始数据提供者所有。

## 🙏 致谢

- [Kaggle](https://www.kaggle.com/) 提供数据集
- [PyTorch](https://pytorch.org/) 深度学习框架
- [torchvision](https://pytorch.org/vision/) 预训练模型

---

**如果这个项目对你有帮助，欢迎 Star ⭐**
