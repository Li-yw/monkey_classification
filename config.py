"""
全局配置文件
集中管理所有超参数和路径设置
"""

# ==================== 数据集配置 ====================
DATA_DIR = "./data"  # 数据集根目录（相对于项目根目录）
TRAIN_DIR = f"{DATA_DIR}/training/training"
VAL_DIR = f"{DATA_DIR}/validation/validation"

# 类别映射（10种猴子物种）
CLASS_NAMES = [
    'mantled_howler',       # 赤吼猴
    'patas_monkey',         # 赤猴
    'bald_uakari',          # 秃猴
    'japanese_macaque',     # 日本猕猴
    'pygmy_marmoset',       # 侏狨
    'white_headed_capuchin',# 白头卷尾猴
    'silvery_marmoset',     # 银狨
    'common_squirrel_monkey',# 松鼠猴
    'black_headed_night_monkey',# 黑头夜猴
    'nilgiri_langur'        # 尼尔吉里叶猴
]
NUM_CLASSES = 10

# ==================== 训练超参数 ====================
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3  # 从零训练
PRETRAINED_LR = 1e-4  # 预训练模型微调
WEIGHT_DECAY = 1e-4   # L2正则化

# ==================== 数据增强参数 ====================
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet 标准化
STD = [0.229, 0.224, 0.225]

# ==================== 模型选择 ====================
AVAILABLE_MODELS = ['simplecnn', 'alexnet', 'vgg16', 'resnet18']

# ==================== 迁移学习开关 ====================
USE_PRETRAINED = True  # True: 使用预训练权重, False: 从零训练
FINETUNE_STRATEGY = 'B'  # 'A': 冻结全部主干, 'B': 冻结浅层, 'C': 全参数解冻

# ==================== 设备配置 ====================
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # Windows兼容性设为0, Linux可设为4

# ==================== 保存路径 ====================
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

# ==================== 学习率调度 ====================
LR_STEP_SIZE = 10  # 每10个epoch调整一次
LR_GAMMA = 0.1     # 学习率衰减系数
