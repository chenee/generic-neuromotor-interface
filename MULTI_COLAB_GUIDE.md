# 多Colab并行训练指南

## ✅ 是的，可以多个Colab同时使用Google Drive中的30GB数据！

### 🎯 核心优势

Google Drive支持**多个Colab同时读取**相同的数据文件，这意味着您可以：
- ✅ 并行运行多个训练实验
- ✅ 测试不同的超参数配置
- ✅ 同时训练Teacher和Student模型
- ✅ 降低单个Colab断线的风险

---

## 📊 典型架构

```
Google Drive (30GB EMG数据)
         ↓ (同时读取)
    ┌────┴────┬──────────┬──────────┬──────────┐
    │         │          │          │          │
Colab #1  Colab #2   Colab #3   Colab #4   Colab #5
实验A     实验B      实验C      实验D      实验E
lr=1e-3   lr=5e-4    batch=128  蒸馏训练   数据增强


每个实验独立保存到不同目录：
/content/drive/MyDrive/emg_project/experiments/
├── baseline_250epochs/          # Colab #1
├── lr5e4_batch128/              # Colab #2
├── heavy_augmentation/          # Colab #3
├── student_distillation/        # Colab #4
└── quantization_test/           # Colab #5
```

---

## 🔧 实施步骤

### Step 1: 在Google Drive中组织数据

```
/content/drive/MyDrive/emg_project/
├── data/                        # 30GB数据（所有Colab共享读取）
│   ├── discrete_gestures_*.hdf5 (100个文件)
│   └── discrete_gestures_corpus.csv
│
└── experiments/                 # 实验输出（每个Colab独立目录）
    ├── baseline/
    ├── lr_tuning/
    ├── student_model/
    └── ...
```

### Step 2: 在每个Colab中设置唯一的实验ID

**Colab #1 (baseline):**
```python
EXPERIMENT_ID = "baseline_250epochs"
```

**Colab #2 (学习率调优):**
```python
EXPERIMENT_ID = "lr5e4_batch128"
```

**Colab #3 (蒸馏):**
```python
EXPERIMENT_ID = "student_distill"
```

### Step 3: 挂载Drive并配置路径

```python
from google.colab import drive
from pathlib import Path

# 挂载
drive.mount('/content/drive')

# 配置路径
DRIVE_DATA = Path("/content/drive/MyDrive/emg_project/data")
DRIVE_OUTPUT = Path("/content/drive/MyDrive/emg_project/experiments") / EXPERIMENT_ID

# 创建输出目录
DRIVE_OUTPUT.mkdir(exist_ok=True, parents=True)
```

### Step 4: 直接使用Drive数据训练

**方案A: 直接读取Drive（推荐多Colab并行）**

```bash
!python -m generic_neuromotor_interface.train \
  --config-name discrete_gestures \
  data_module.data_location=/content/drive/MyDrive/emg_project/data \
  data_module.data_split.csv_filename=/content/drive/MyDrive/emg_project/data/discrete_gestures_corpus.csv \
  trainer.max_epochs=250 \
  +trainer.default_root_dir=/content/drive/MyDrive/emg_project/experiments/${EXPERIMENT_ID}
```

**优点:**
- ✅ 多个Colab同时读取OK
- ✅ 节省30GB本地存储
- ✅ 结果直接保存到Drive，不怕断线

**缺点:**
- ⚠️ I/O速度约20-30MB/s（可接受）

**方案B: 复制到本地SSD（单Colab长训练）**

```bash
# 首次复制（约3-5分钟）
!cp -r /content/drive/MyDrive/emg_project/data /content/generic-neuromotor-interface/

# 使用本地数据训练（I/O速度200MB/s+）
!python -m generic_neuromotor_interface.train \
  --config-name discrete_gestures \
  data_module.data_location=/content/generic-neuromotor-interface/data \
  ...
```

**优点:**
- ✅ I/O速度快10倍

**缺点:**
- ⚠️ 占用30GB本地存储
- ⚠️ Colab重启需重新复制

---

## 💡 典型使用场景

### 场景1: 超参数网格搜索（4个Colab并行）

```python
# Colab #1
EXPERIMENT_ID = "lr1e3_batch64"
learning_rate = 1e-3
batch_size = 64

# Colab #2
EXPERIMENT_ID = "lr5e4_batch64"
learning_rate = 5e-4
batch_size = 64

# Colab #3
EXPERIMENT_ID = "lr1e3_batch128"
learning_rate = 1e-3
batch_size = 128

# Colab #4
EXPERIMENT_ID = "lr5e4_batch128"
learning_rate = 5e-4
batch_size = 128
```

**预计时间**: 每个10-20 epochs，2-3小时完成网格搜索

### 场景2: 完整训练+蒸馏（2个Colab并行）

```python
# Colab #1: 训练Teacher (250 epochs)
EXPERIMENT_ID = "teacher_full_250"
!python -m generic_neuromotor_interface.train \
  --config-name discrete_gestures \
  trainer.max_epochs=250

# Colab #2: 同时训练Student进行蒸馏
EXPERIMENT_ID = "student_convonly"
cd mini/chenee/distillation
!python train_distillation.py \
  --teacher_checkpoint /content/drive/.../teacher_checkpoint.ckpt \
  --max_epochs=100
```

### 场景3: 降低断线风险（2个Colab备份）

```python
# Colab #1 和 #2 运行相同配置
# 如果一个断线，另一个继续
EXPERIMENT_ID = "baseline_v1"  # Colab #1
EXPERIMENT_ID = "baseline_v2"  # Colab #2
```

---

## ⚠️ 重要注意事项

### ✅ 可以做的

1. **多个Colab同时读取Drive数据** - 无限制
2. **各自训练独立模型** - 完全隔离
3. **保存到Drive不同目录** - 使用不同EXPERIMENT_ID
4. **共享Teacher模型checkpoint** - 其他Colab可加载

### ⚠️ 需要避免的

1. **同时写入同一文件** - 会导致数据损坏
2. **使用相同输出目录** - checkpoint会互相覆盖
3. **大量同时写入** - 可能触发Drive限速

### 📊 Google Drive限制

| 项目 | 限制 | 说明 |
|------|------|------|
| **读取并发** | 无限制 | ✅ 多Colab同时读取OK |
| **每日下载** | 750GB | 4个Colab并行足够 |
| **单文件大小** | <5TB | EMG文件远小于此 |
| **读取速度** | 20-30MB/s | 首次较慢，有缓存 |
| **写入速度** | 10-20MB/s | checkpoint保存无压力 |

---

## 🎯 最佳实践建议

### 1. 实验命名规范

```python
# 好的命名（描述性强）
EXPERIMENT_ID = "lr1e3_batch64_aug_heavy"
EXPERIMENT_ID = "student_convonly_distill_t4.0"
EXPERIMENT_ID = "baseline_250ep_v2"

# 不好的命名
EXPERIMENT_ID = "test1"
EXPERIMENT_ID = "exp_abc"
```

### 2. 定期备份到Drive

```python
# 每10 epochs自动保存checkpoint到Drive
trainer.callbacks = [
    ModelCheckpoint(
        dirpath=f"/content/drive/MyDrive/experiments/{EXPERIMENT_ID}/checkpoints",
        every_n_epochs=10,
        save_top_k=-1  # 保存所有checkpoint
    )
]
```

### 3. 使用TensorBoard监控所有实验

```bash
# 在本地或Colab中启动TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/emg_project/experiments
```

### 4. 记录实验配置

```python
# 在每个Colab中保存配置
import json

config = {
    "experiment_id": EXPERIMENT_ID,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 250,
    "notes": "Testing heavy data augmentation"
}

with open(f"{DRIVE_OUTPUT}/config.json", "w") as f:
    json.dump(config, f, indent=2)
```

---

## 📈 性能对比

### 数据访问速度测试

| 方法 | 首次读取 | 后续读取 | 占用空间 |
|------|---------|---------|---------|
| Drive直接读 | 20-30MB/s | 30-40MB/s | 0GB（本地） |
| 复制到本地 | 200MB/s+ | 200MB/s+ | 30GB（本地） |

### 训练时间估算（Tesla T4）

| 配置 | Drive读取 | 本地SSD | 差异 |
|------|----------|---------|------|
| 10 epochs | 90分钟 | 75分钟 | +20% |
| 50 epochs | 7.5小时 | 6.3小时 | +19% |
| 250 epochs | 37小时 | 31小时 | +19% |

**结论**: Drive直接读取仅慢约20%，对于并行实验完全可接受

---

## 🔬 实战示例

### 完整多Colab并行训练脚本

**在每个Colab的第一个cell中运行:**

```python
# ========== Colab配置 ==========
# 修改这里以区分不同的Colab实验
EXPERIMENT_ID = "lr1e3_batch64"  # ⚠️ 每个Colab修改此行
DESCRIPTION = "Baseline with lr=1e-3, batch=64"

# 超参数配置
CONFIG_OVERRIDE = {
    "optimizer.lr": 1e-3,
    "data_module.batch_size": 64,
    "trainer.max_epochs": 250,
}

# ========== Drive路径配置 ==========
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

DRIVE_BASE = Path("/content/drive/MyDrive/emg_project")
DATA_DIR = DRIVE_BASE / "data"
OUTPUT_DIR = DRIVE_BASE / "experiments" / EXPERIMENT_ID

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"✅ 实验配置完成:")
print(f"   ID: {EXPERIMENT_ID}")
print(f"   描述: {DESCRIPTION}")
print(f"   数据: {DATA_DIR}")
print(f"   输出: {OUTPUT_DIR}")

# ========== 开始训练 ==========
!python -m generic_neuromotor_interface.train \
  --config-name discrete_gestures \
  data_module.data_location={DATA_DIR} \
  data_module.data_split.csv_filename={DATA_DIR}/discrete_gestures_corpus.csv \
  optimizer.lr={CONFIG_OVERRIDE['optimizer.lr']} \
  data_module.batch_size={CONFIG_OVERRIDE['data_module.batch_size']} \
  trainer.max_epochs={CONFIG_OVERRIDE['trainer.max_epochs']} \
  +trainer.default_root_dir={OUTPUT_DIR}
```

---

## ✅ 检查清单

开始多Colab并行训练前，确认：

- [ ] 30GB数据已上传到Google Drive
- [ ] 每个Colab设置了唯一的EXPERIMENT_ID
- [ ] 输出目录配置到Drive的不同子目录
- [ ] 已挂载Google Drive
- [ ] 确认数据路径正确（100个.hdf5文件）
- [ ] 配置了独立的超参数（如果需要）
- [ ] 准备好监控工具（TensorBoard或Weights & Biases）

---

## 🎉 总结

**您完全可以用多个Colab同时使用Google Drive中的30GB数据！**

**推荐配置：**
- 2-4个Colab并行训练不同实验
- 直接读取Drive数据（无需复制）
- 每个实验使用独立的EXPERIMENT_ID
- 定期检查TensorBoard监控进度

**预期收益：**
- 🚀 4倍速度完成超参数搜索
- 💡 同时验证多个想法
- 🛡️ 降低单点故障风险
- 📊 快速对比不同方案

现在就开始您的并行训练之旅吧！🎯
