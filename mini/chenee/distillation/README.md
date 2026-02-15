# 离散手势识别 - 知识蒸馏

用轻量级Student模型（~60万参数）蒸馏Meta的Teacher大模型（~650万参数），实现**10倍模型压缩**。

## 📋 项目结构

```
distillation/
├── student_network.py          # Student小模型定义（~60万参数）
├── distillation_module.py      # 蒸馏训练Lightning模块
├── train_distillation.py       # 主训练脚本
├── config_distillation.yaml    # 训练配置文件
└── README.md                   # 本文件
```

## 🎯 核心思想

**知识蒸馏**：让小模型（Student）学习大模型（Teacher）的"知识"，而不是从头训练。

### 损失函数设计

```
总损失 = α × 蒸馏损失 + (1-α) × 任务损失

蒸馏损失 = KL_divergence(Student概率, Teacher概率) × T²
任务损失 = BCE(Student输出, 真实标签)
```

- **温度参数 T**：softmax平滑度（推荐2-5）
- **α权重**：蒸馏/任务平衡（推荐0.5）

## 🏗️ 模型对比

### 标准模型

| 特性 | Teacher (Meta大模型) | Student (轻量级) | 压缩比 |
|------|---------------------|-----------------|-------|
| **Conv通道** | 512 | 128 | 4× |
| **LSTM层数** | 3层 | 2层 | - |
| **LSTM隐藏单元** | 512 | 256 | 2× |
| **总参数量** | ~650万 | ~60万 | **10%** |
| **推理速度** | 基准 | ~3-4×加速 | - |

### 🆕 超轻量级模型（ESP32适配）

| 模型 | 参数量 | FP32 | INT8 | 推荐设备 |
|------|--------|------|------|---------|
| **ConvOnly** | 8万 | 320KB | **80KB** | ✅ ESP32 |
| **TinyStudent** | 15万 | 600KB | 150KB | ESP32-S3 |
| **GRUStudent** | 40万 | 1.6MB | 400KB | 树莓派 |

**ESP32部署**:
- 👉 模型对比: [MODEL_COMPARISON.md](MODEL_COMPARISON.md)
- 👉 完整指南: [ESP32_DEPLOYMENT.md](ESP32_DEPLOYMENT.md)
- 👉 模型源码: [ultra_light_models.py](ultra_light_models.py)

### 参数分解

**Teacher模型**（6,482,953参数）:
- Conv1d: ~428K
- LSTM: ~5.7M  
- 投影层: ~4.6K

**Student模型**（~600K参数）:
- Conv1d: ~27K
- LSTM: ~560K
- 投影层: ~2.3K

## 🚀 快速开始

### 1. 准备Teacher模型

确保你已经训练好Teacher模型：

```bash
# 应该有这个文件
ls ../../../../../logs/best_discrete_gestures.pt
```

### 2. 运行蒸馏训练

```bash
python train_distillation.py \
    --teacher_checkpoint ../../../../../logs/best_discrete_gestures.pt \
    --data_dir /path/to/emg_data \
    --split_csv /path/to/split.csv \
    --output_dir ./student_models \
    --batch_size 16 \
    --max_epochs 100 \
    --temperature 3.0 \
    --alpha 0.5
```

### 3. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--teacher_checkpoint` | **必需** | Teacher模型权重路径 |
| `--data_dir` | **必需** | EMG数据目录 |
| `--split_csv` | **必需** | 数据划分CSV |
| `--output_dir` | `./distillation_output` | 模型保存目录 |
| `--batch_size` | 16 | 批次大小 |
| `--max_epochs` | 100 | 最大训练轮数 |
| `--learning_rate` | 1e-3 | 学习率 |
| `--temperature` | 3.0 | 蒸馏温度（2-5推荐） |
| `--alpha` | 0.5 | 蒸馏损失权重（0.3-0.7） |
| `--gpus` | 1 | GPU数量 |

## 📊 训练监控

训练过程会记录以下指标：

- `train_loss` - 总训练损失
- `train_distill_loss` - KL散度蒸馏损失
- `train_task_loss` - BCE任务损失
- `val_loss` - 验证损失
- `val_accuracy` - 验证准确率

使用TensorBoard查看：

```bash
tensorboard --logdir distillation_output/lightning_logs
```

## 🔧 超参数调优

### Temperature（温度）

- **低温（1-2）**：Student更专注学习Teacher的确定性预测
- **中温（3-4）**：平衡，推荐起点
- **高温（5-7）**：Student学习Teacher的uncertainty，泛化能力更强

### Alpha（蒸馏权重）

- **低α（0.2-0.3）**：更依赖真实标签，适合Teacher不太准确
- **中α（0.4-0.6）**：平衡，推荐起点
- **高α（0.7-0.9）**：更依赖Teacher软标签，适合Teacher很准确

### 典型配置组合

| 场景 | Temperature | Alpha | 说明 |
|------|------------|-------|------|
| **保守策略** | 2.0 | 0.3 | Teacher不够准确时 |
| **平衡策略** | 3.0 | 0.5 | 推荐默认配置 |
| **激进策略** | 4.0 | 0.7 | Teacher非常准确时 |

## 📈 预期效果

根据经验，蒸馏后的Student模型通常能达到：

- **准确率**：Teacher的90-95%
- **CLER指标**：Teacher的85-90%
- **推理速度**：3-4倍加速
- **内存占用**：~10%

## 🧪 测试Student模型

训练完成后测试：

```python
import torch
from student_network import StudentDiscreteGesturesArchitecture

# 加载模型
student = StudentDiscreteGesturesArchitecture()
student.load_state_dict(torch.load('student_models/student_final.pt'))
student.eval()

# 测试推理
dummy_emg = torch.randn(1, 16, 1000)  # (batch=1, channels=16, time=1000)
output = student(dummy_emg)
print(f"输出shape: {output.shape}")  # (1, 9, downsampled_time)
```

## 🎓 进阶技巧

### 1. 渐进式蒸馏

如果直接蒸馏效果不好，可以尝试：

```
Teacher (650万) → Medium (200万) → Student (60万)
```

### 2. 特征蒸馏

在`distillation_module.py`中添加中间层特征匹配：

```python
# 蒸馏LSTM隐藏状态
student_hidden = student.lstm_output
teacher_hidden = teacher.lstm_output
feature_loss = F.mse_loss(student_hidden, teacher_hidden)
```

### 3. 数据增强

蒸馏时可以使用更强的数据增强，因为Teacher提供了稳定的监督信号。

## 🐛 常见问题

### Q1: 训练不收敛怎么办？

- 降低学习率（试试5e-4）
- 增大batch_size
- 减小temperature（试试2.0）
- 降低alpha（试试0.3，更依赖真实标签）

### Q2: Student准确率太低？

- 增大Student模型（conv=256, lstm=384）
- 提高temperature（试试4.0）
- 确保Teacher模型本身够准确
- 延长训练轮数

### Q3: 过拟合了？

- 增大dropout（从0.1到0.2）
- 减少max_epochs
- 使用early stopping（已内置）

## 📚 参考文献

1. Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
2. Meta's Generic Neuromotor Interface paper

## 📝 TODO

- [ ] 添加量化感知训练（INT8部署）
- [ ] 实现剪枝+蒸馏组合压缩
- [ ] 导出ONNX/TFLite格式
- [ ] 添加边缘设备推理benchmark

## 💡 贡献

欢迎提交Issue和PR！
