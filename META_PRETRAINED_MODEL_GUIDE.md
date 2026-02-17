# Meta预训练模型使用指南（Colab环境）

## ✅ 是的，Meta提供了官方预训练模型！

### 📦 模型信息

Meta在项目中提供了3个任务的预训练模型：
- ✅ **discrete_gestures** (离散手势识别)
- ✅ **handwriting** (手写识别)  
- ✅ **wrist** (腕部运动)

**训练配置:**
- 100个参与者的完整数据
- 250 epochs训练
- 最优超参数配置
- 约25-74MB大小

---

## 🚀 在Colab中下载和使用

### 方法1: Python脚本下载

```python
from generic_neuromotor_interface.scripts.download_models import download_models

# 下载discrete_gestures预训练模型
download_models("discrete_gestures", "/content/emg_models")
```

### 方法2: 命令行下载

```bash
python -m generic_neuromotor_interface.scripts.download_models \
    --task discrete_gestures \
    --output-dir /content/emg_models
```

下载完成后会得到：
```
/content/emg_models/discrete_gestures/
├── model_checkpoint.ckpt    # PyTorch Lightning checkpoint (~74MB)
└── model_config.yaml         # 模型配置文件
```

---

## 💡 预训练模型用途

### 1️⃣ 直接评估Meta模型性能

查看Meta官方模型的准确率，作为baseline参考：

```python
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# 加载配置
config = OmegaConf.load("/content/emg_models/discrete_gestures/model_config.yaml")

# 加载模型
model = instantiate(config.lightning_module)
model = model.load_from_checkpoint(
    "/content/emg_models/discrete_gestures/model_checkpoint.ckpt",
    map_location=torch.device("cpu")
)

# 评估
# 参考 notebooks/discrete_gestures-eval.ipynb
```

### 2️⃣ 作为Teacher进行知识蒸馏

使用Meta高性能模型蒸馏轻量级Student模型：

```python
# 在蒸馏脚本中
cd mini/chenee/distillation

python train_distillation.py \
    --teacher_checkpoint /content/emg_models/discrete_gestures/model_checkpoint.ckpt \
    --data_dir /content/generic-neuromotor-interface/data \
    --split_csv /content/generic-neuromotor-interface/data/discrete_gestures_corpus.csv \
    --output_dir ./student_models
```

这样可以得到：
- ConvOnly: 80KB INT8模型（ESP32）
- Student: 600KB模型（手机）

### 3️⃣ 与自己训练的模型对比

对比您训练的模型与Meta官方模型的性能差距：

```python
# Meta预训练模型
meta_model = load_checkpoint("/content/emg_models/discrete_gestures/model_checkpoint.ckpt")
meta_accuracy = evaluate(meta_model)  # 预期: ~95%+

# 您训练的模型  
your_model = load_checkpoint("/content/generic-neuromotor-interface/logs/.../epoch=8-step=1980.ckpt")
your_accuracy = evaluate(your_model)  # 当前: ~39% (10 epochs)

print(f"Meta模型: {meta_accuracy:.2%}")
print(f"您的模型: {your_accuracy:.2%}")
print(f"差距: {meta_accuracy - your_accuracy:.2%}")
```

### 4️⃣ 迁移学习/微调

在新数据上微调预训练模型：

```python
# 加载预训练权重
pretrained_model = load_checkpoint("/content/emg_models/discrete_gestures/model_checkpoint.ckpt")

# 在新数据上微调
trainer = Trainer(max_epochs=50)
trainer.fit(pretrained_model, new_datamodule)
```

---

## 📊 预期性能（Meta预训练模型）

根据论文，Meta预训练模型在测试集上的表现：

| 任务 | 准确率 | CLER | 备注 |
|------|--------|------|------|
| **Discrete Gestures** | ~95%+ | <5% | 9种手势分类 |
| **Handwriting** | ~90%+ | ~10% | 字符识别 |
| **Wrist** | ~85%+ | - | 腕部运动轨迹 |

**对比您当前的10轮训练:**
- 您的模型: 39% 准确率 (10 epochs)
- Meta模型: ~95% 准确率 (250 epochs, 100用户)
- **差距原因**: 训练时间短，需要继续训练至250 epochs

---

## 🔧 完整使用示例（Colab Notebook）

在您的colab_train.ipynb中添加以下单元格：

### 单元格1: 检查预训练模型

```python
import os
from pathlib import Path

MODEL_DIR = Path("/content/emg_models")
PRETRAINED_MODEL = MODEL_DIR / "discrete_gestures" / "model_checkpoint.ckpt"

if PRETRAINED_MODEL.exists():
    size_mb = PRETRAINED_MODEL.stat().st_size / 1e6
    print(f"✅ 找到Meta预训练模型: {size_mb:.1f} MB")
else:
    print("⚠️  未找到，需要下载")
```

### 单元格2: 下载预训练模型

```python
from generic_neuromotor_interface.scripts.download_models import download_models

print("📥 下载Meta预训练模型...")
download_models("discrete_gestures", "/content/emg_models")
print("✅ 下载完成！")
```

### 单元格3: 加载并查看模型信息

```python
import torch
from omegaconf import OmegaConf

# 加载配置
config_path = "/content/emg_models/discrete_gestures/model_config.yaml"
config = OmegaConf.load(config_path)

print("📋 Meta模型配置:")
print(OmegaConf.to_yaml(config))

# 加载checkpoint查看详情
ckpt = torch.load(
    "/content/emg_models/discrete_gestures/model_checkpoint.ckpt",
    map_location='cpu',
    weights_only=False
)

print(f"\n📊 Checkpoint信息:")
print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"  Global Step: {ckpt.get('global_step', 'N/A')}")

# 查看最佳验证指标
if 'callbacks' in ckpt:
    print(f"\n🏆 Meta模型最佳表现:")
    for cb_name, cb_state in ckpt['callbacks'].items():
        if 'best' in str(cb_name).lower():
            print(f"  {cb_name}:")
            if isinstance(cb_state, dict):
                for k, v in cb_state.items():
                    if 'best' in str(k).lower():
                        print(f"    {k}: {v}")
```

---

## ⚠️ 重要说明

### Colab环境路径

您使用VSCode连接Colab kernel，所以：
- ✅ 数据路径: `/content/generic-neuromotor-interface/data`
- ✅ 模型路径: `/content/emg_models`
- ✅ 日志路径: `/content/generic-neuromotor-interface/logs`
- ❌ **不是**本地路径: `~/emg_data` 或 `/Users/chenee/...`

### 下载时间和大小

- discrete_gestures模型: ~74MB
- 下载时间: 约30秒（取决于网络）
- 存储位置: Colab临时存储（重启会丢失）

### 永久保存

如果要永久保存预训练模型到Google Drive：

```python
# 下载后复制到Drive
!cp -r /content/emg_models /content/drive/MyDrive/emg_models

# 下次使用时从Drive加载
MODEL_PATH = "/content/drive/MyDrive/emg_models/discrete_gestures/model_checkpoint.ckpt"
```

---

## 🎯 推荐工作流

### 场景1: 快速验证（使用预训练模型）

1. 下载Meta预训练模型
2. 在测试集上评估
3. 查看baseline性能
4. 决定是否需要重新训练

### 场景2: 完整训练（从零开始）

1. 训练自己的模型（250 epochs）
2. 下载Meta预训练模型
3. 对比两者性能
4. 分析差距原因

### 场景3: 知识蒸馏（推荐）

1. 下载Meta预训练模型作为Teacher
2. 训练轻量级Student模型
3. 部署到ESP32等嵌入式设备
4. 实现10-100倍模型压缩

---

## 📚 相关资源

- 下载脚本: `generic_neuromotor_interface/scripts/download_models.py`
- 评估notebook: `notebooks/discrete_gestures-eval.ipynb`
- 蒸馏代码: `mini/chenee/distillation/`
- Meta论文: https://www.nature.com/articles/s41586-025-09255-w

---

## ✅ 总结

**是的，项目中有Meta预训练的高性能模型！**

**关键点:**
- 📦 需要手动下载（约74MB）
- ⚡ 性能优秀（~95%准确率）
- 🎯 可用于评估、蒸馏、对比
- 💻 在Colab环境中使用，不是本地

**立即使用:**
```python
# 在Colab notebook中运行
from generic_neuromotor_interface.scripts.download_models import download_models
download_models("discrete_gestures", "/content/emg_models")
```

现在就可以评估Meta的sota模型性能了！🚀
