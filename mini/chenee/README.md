# Chenee的工作目录

这是chenee的个人工作区，位于 `mini/chenee/` 目录下。

## 📂 目录结构

```
mini/chenee/
└── distillation/          # 知识蒸馏项目
    ├── student_network.py           # Student网络（~60万参数）
    ├── ultra_light_models.py        # 🆕 超轻量级模型（ESP32适配）
    ├── quantization_utils.py        # 🆕 INT8量化工具
    ├── distillation_module.py       # 蒸馏训练模块
    ├── train_distillation.py        # 训练脚本
    ├── test_setup.py               # 验证脚本
    ├── README.md                   # 完整文档
    ├── MODEL_COMPARISON.md          # 🆕 模型对比说明
    ├── ESP32_DEPLOYMENT.md          # 🆕 ESP32部署指南
    ├── USAGE.py                    # 使用示例
    ├── config_distillation.yaml     # 配置
    ├── quickstart.sh               # 快速开始
    └── __init__.py                 # 模块初始化
```

## 🎯 distillation 项目

**目标**: 将Meta的离散手势识别大模型（650万参数）蒸馏到轻量级Student模型（60万参数）

**压缩比**: 10%（10倍压缩）

**🆕 ESP32超轻量级方案**:
- **ConvOnly**: 8万参数，INT8仅80KB，适合ESP32
- **TinyStudent**: 15万参数，INT8约150KB，ESP32-S3
- **GRUStudent**: 40万参数，INT8约400KB，树莓派

**预期效果**:
- 准确率达到Teacher的90-95%
### 标准蒸馏训练

- 推理速度提升3-4倍
- 适合边缘设备部署

## 🚀 快速开始

```bash
# 进入项目目录
cd distillation

# 验证环境
python3 test_setup.py

# 开始训练
python3 train_distillation.py \
    --teacher_checkpoint ../../../../../logs/best_discrete_gestures.pt \
    --data_dir /path/to/emg_data \
    --split_csv /path/to/split.csv
### 🆕 ESP32超轻量级训练

```bash
# 1. 查看模型对比
cd distillation
python3 ultra_light_models.py

# 2. 训练ConvOnly模型（80KB INT8）
# 修改train_distillation.py使用ConvOnlyStudentArchitecture
python3 train_distillation.py \
    --teacher_checkpoint ../../../../../logs/best_discrete_gestures.pt \
    --data_dir /path/to/emg_data \
    --split_csv /path/to/split.csv \
    --output_dir ./esp32_models

# 3. 量化为INT8
python3 quantization_utils.py

# 4. 部署到ESP32
# 查看完整指南: ESP32_DEPLOYMENT.md
```

详细文档请查看 `distillation/README.md` 和 `distillation/ESP32_DEPLOYMENT

详细文档请查看 `distillation/README.md`

## 📊 与其他目录的关系

```
mini/
├── LLZ/              # LLZ的工作区（TFLite模型分析）
└── chenee/           # Chenee的工作区
    └── distillation/ # 知识蒸馏项目
```

## 📝 备注

- 所有路径都已配置为相对路径
- Teacher模型位置: `../../../../../logs/best_discrete_gestures.pt`
- 输出默认保存到: `distillation/student_models/`
