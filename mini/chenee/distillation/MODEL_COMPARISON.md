# ESP32模型快速对比

## 🎯 三种超轻量级方案

```
┌─────────────────────────────────────────────────────────────┐
│                     模型规模对比图                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Teacher   ████████████████████████████████ 650万 (100%)   │
│                                                             │
│  Student   ████                              60万 (9%)     │
│                                                             │
│  TinyStudent █                               15万 (2.3%)   │
│                                                             │
│  ConvOnly  ▌                                 8万  (1.2%)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📊 详细对比

### 1. ConvOnly - 全卷积网络 ⭐推荐ESP32

```python
结构: Conv(16→64) → Conv(64→64) → Conv(64→32) → Linear(32→9)
```

**参数**: 8万  
**大小**: FP32 320KB → INT8 **80KB**  
**特点**:
- ✅ 最小、最快
- ✅ 无LSTM，无状态，内存低
- ✅ 易量化，精度损失<2%
- ⚠️ 无法建模长期依赖
- ⚠️ 精度约为Teacher的75-80%

**适合**: ESP32 (520KB RAM)

**创建方式**:
```python
from ultra_light_models import ConvOnlyStudentArchitecture
model = ConvOnlyStudentArchitecture(hidden_channels=64)
```

---

### 2. TinyStudent - 超轻LSTM

```python
结构: Conv(16→64, stride=10) → 1×LSTM(128) → Linear(128→9)
```

**参数**: 15万  
**大小**: FP32 600KB → INT8 **150KB**  
**特点**:
- ✅ 有LSTM，能建模时序
- ✅ 精度约为Teacher的82-85%
- ⚠️ 需要更多RAM
- ⚠️ 推理稍慢

**适合**: ESP32-S3 (更大RAM) 或树莓派Pico

**创建方式**:
```python
from ultra_light_models import TinyStudentArchitecture
model = TinyStudentArchitecture(
    conv_output_channels=64,
    lstm_hidden_size=128,
    lstm_num_layers=1
)
```

---

### 3. GRUStudent - GRU替代LSTM

```python
结构: Conv(16→128, stride=10) → 2×GRU(256) → Linear(256→9)
```

**参数**: 40万  
**大小**: FP32 1.6MB → INT8 **400KB**  
**特点**:
- ✅ GRU比LSTM少25%参数
- ✅ 精度约为Teacher的88-90%
- ⚠️ ESP32可能内存不足

**适合**: 树莓派、Android设备

**创建方式**:
```python
from ultra_light_models import GRUStudentArchitecture
model = GRUStudentArchitecture()
```

---

## 🚀 快速开始

### 1. 查看所有模型对比
```bash
cd distillation
python3 ultra_light_models.py
```

### 2. 训练ConvOnly模型（ESP32推荐）

修改 `train_distillation.py`:
```python
# 在create_student_model()函数中
from ultra_light_models import ConvOnlyStudentArchitecture

student = ConvOnlyStudentArchitecture(
    input_channels=16,
    hidden_channels=64,  # 可调: 32/64/128
    output_channels=9,
)
```

然后训练:
```bash
python3 train_distillation.py \
    --teacher_checkpoint ../../../../../logs/best_discrete_gestures.pt \
    --data_dir /path/to/emg_data \
    --split_csv /path/to/split.csv \
    --output_dir ./esp32_models \
    --temperature 4.0 \
    --alpha 0.6
```

### 3. 量化为INT8
```bash
python3 quantization_utils.py
```

### 4. 部署到ESP32
查看完整指南: [ESP32_DEPLOYMENT.md](ESP32_DEPLOYMENT.md)

---

## 💡 如何选择？

| 需求 | 推荐模型 | 说明 |
|------|---------|------|
| **ESP32原版** | ConvOnly | 最小最快，80KB |
| **ESP32-S3** | TinyStudent | 精度更好，150KB |
| **树莓派Pico** | TinyStudent 或 GRUStudent | - |
| **Android/iOS** | 标准Student | 60万参数 |
| **云端/PC** | Teacher | 最高精度 |

---

## 🎯 精度预期

```
模型层级            精度        参数量      INT8大小
================   ======      =======     ========
Teacher            95%         650万       -
Student            90%         60万        600KB
GRUStudent         88%         40万        400KB
TinyStudent        85%         15万        150KB
ConvOnly           78%         8万         80KB
```

**INT8量化损失**: 通常 <2%

---

## 📝 更多资源

- **完整ESP32部署**: [ESP32_DEPLOYMENT.md](ESP32_DEPLOYMENT.md)
- **模型源码**: [ultra_light_models.py](ultra_light_models.py)
- **量化工具**: [quantization_utils.py](quantization_utils.py)
- **使用示例**: [USAGE.py](USAGE.py)
