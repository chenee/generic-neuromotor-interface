# mytrain.ipynb - EMG手势识别学习笔记

## 📚 简介

这是一个循序渐进的Jupyter Notebook，用于学习基于EMG（肌电信号）的手势识别系统。

## ✅ 状态

**已完成调试** - 所有组件已验证正常工作（详见 `NOTEBOOK_DEBUG_REPORT.txt`）

## 🚀 快速开始

### 1. 环境要求

```bash
Python 3.12+
PyTorch 2.2+
pytorch-lightning 1.8+
h5py, pandas, matplotlib, numpy
```

### 2. 数据准备

确保数据位于正确位置：
```
~/emg_data/
  ├── discrete_gestures_user_000_dataset_000.hdf5
  ├── discrete_gestures_user_001_dataset_000.hdf5
  └── discrete_gestures_user_002_dataset_000.hdf5
```

### 3. 运行Notebook

1. 打开 `mytrain.ipynb`
2. **重要**: 从头按顺序执行所有cells（推荐使用 "Run All"）
3. 首次训练建议修改 `MAX_EPOCHS = 1` 进行快速测试

## 📖 Notebook结构

### 第一部分：数据理解（Cells 1-6）
- ✅ Cell 1: 导入必要的库
- ✅ Cell 2: 配置参数（路径、批量大小等）
- ✅ Cell 3: 定义手势类型（9种手势）
- ✅ Cell 4: 加载数据集划分
- ✅ Cell 5: **数据探索** - 查看HDF5文件结构
- ✅ Cell 6: **数据可视化** - EMG信号和事件时间线

### 第二部分：数据处理（Cells 7-11）
- ✅ Cell 7: EmgRecording类 - 封装单个录制文件
- ✅ Cell 8: 数据变换 - 生成手势标签脉冲
- ✅ Cell 9: 结果可视化 - 查看变换后的标签
- ✅ Cell 10: WindowedEmgDataset - 滑动窗口数据集
- ✅ Cell 11: DataModule - 批量数据加载

### 第三部分：模型构建（Cells 12-14）
- ✅ Cell 12: 数据增强 - 通道旋转
- ✅ Cell 13: TCN网络架构 - 6.3M参数
- ✅ Cell 14: Lightning模块 - 训练框架

### 第四部分：训练流程（Cells 15-17）
- ✅ Cell 15: 状态掩码生成器
- ✅ Cell 16: 完整训练循环配置
- ⏸️ Cell 17: 实际训练执行（未测试，需用户运行）

## 🎯 数据格式说明

### HDF5文件结构
```
discrete_gestures_user_XXX.hdf5
├── data (structured array)
│   ├── emg: (N, 16) - 16通道EMG信号
│   └── time: (N,) - 绝对时间戳
├── prompts (DataFrame)
│   ├── name: 手势名称
│   └── time: 事件时间
└── stages (DataFrame) - 录制阶段信息
```

### 手势类型
1. `index_press` / `index_release` - 食指按下/释放
2. `middle_press` / `middle_release` - 中指按下/释放
3. `thumb_up` / `thumb_down` - 拇指上/下
4. `thumb_in` / `thumb_out` - 拇指内/外
5. `thumb_click` - 拇指点击

## 📊 关键参数

```python
SAMPLE_RATE = 2000      # 采样率 2kHz
WINDOW_SIZE = 512       # 窗口大小 256ms
WINDOW_STRIDE = 64      # 步长 32ms (88%重叠)
BATCH_SIZE = 64         # 批量大小
MAX_EPOCHS = 20         # 训练轮数
LEARNING_RATE = 1e-3    # 学习率
```

## ⚠️ 注意事项

1. **必须按顺序执行**: 后续cells依赖前面的变量
2. **内存需求**: 全数据集训练需要~4-8GB RAM
3. **GPU推荐**: 使用GPU可大幅加速（当前默认CPU）
4. **中文显示**: matplotlib可能显示中文为方框（不影响功能）

## 🐛 常见问题

### Q: KeyError: 'emg' 或类似错误
**A**: 确保按顺序执行所有cells，不要跳过任何cell

### Q: 找不到数据文件
**A**: 检查 `DATA_DIR` 路径是否正确，数据文件是否存在

### Q: 训练太慢
**A**: 
- 方案1: 修改 `MAX_EPOCHS = 1` 进行快速测试
- 方案2: 使用GPU（如果可用）
- 方案3: 减少 `BATCH_SIZE`

### Q: Out of Memory
**A**: 
- 减少 `BATCH_SIZE`
- 减少 `WINDOW_SIZE`
- 只使用部分数据文件进行测试

## 📝 修改记录

- 2024: 修复数据加载逻辑，适配pandas HDFStore格式
- 2024: 更新手势类型枚举，匹配实际数据标签
- 2024: 修复EmgRecording类，支持结构化数组
- 2024: 修复状态掩码生成器，仅对index/middle应用
- 2024: 完成端到端调试验证

## 📚 相关文档

- 详细调试报告: `NOTEBOOK_DEBUG_REPORT.txt`
- 原始训练脚本: `mytrain.py`
- 验证脚本: `validate_fixes.py`

## 👥 使用建议

### 初学者
1. 仔细阅读每个cell的注释
2. 运行到Cell 11后暂停，理解数据流程
3. 查看所有可视化结果
4. 再继续运行模型相关的cells

### 实验者
1. 修改超参数（Cell 2）
2. 尝试不同的网络架构（Cell 13）
3. 调整数据增强策略（Cell 12）
4. 对比不同配置的训练效果

### 开发者
1. 理解EmgRecording的设计（Cell 7）
2. 学习WindowedEmgDataset的实现（Cell 10）
3. 研究TCN网络结构（Cell 13）
4. 自定义损失函数和指标

---

**Happy Learning! 🎉**
