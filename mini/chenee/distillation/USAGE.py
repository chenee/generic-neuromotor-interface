"""
蒸馏训练使用示例

演示如何使用知识蒸馏框架
"""

# ============================================================
# 示例1：命令行训练（推荐）
# ============================================================

"""
最简单的用法：

python train_distillation.py \
    --teacher_checkpoint ../../../../../logs/best_discrete_gestures.pt \
    --data_dir /path/to/emg_data \
    --split_csv /path/to/split.csv

这会使用默认配置：
- 批次大小: 16
- 最大轮数: 100
- 温度: 3.0
- Alpha: 0.5
"""


# ============================================================
# 示例2：自定义参数训练
# ============================================================

"""
调整蒸馏参数（激进策略，更依赖Teacher）：

python train_distillation.py \
    --teacher_checkpoint ../../../../../logs/best_discrete_gestures.pt \
    --data_dir /path/to/emg_data \
    --split_csv /path/to/split.csv \
    --temperature 4.0 \
    --alpha 0.7 \
    --max_epochs 150 \
    --batch_size 32
"""


# ============================================================
# 示例3：Python脚本调用
# ============================================================

if __name__ == "__main__":
    from train_distillation import train_distillation

    # 配置参数
    config = {
        "teacher_checkpoint": "../../../../../logs/best_discrete_gestures.pt",
        "data_dir": "/path/to/emg_data",
        "split_csv": "/path/to/split.csv",
        "output_dir": "./my_student_models",
        "batch_size": 16,
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "temperature": 3.0,
        "alpha": 0.5,
        "gpus": 1,
    }

    # 开始训练
    model, trainer = train_distillation(**config)

    print("训练完成！")


# ============================================================
# 示例4：推理使用Student模型
# ============================================================

"""
训练完成后使用Student模型：
"""

def inference_example():
    import torch
    from student_network import StudentDiscreteGesturesArchitecture

    # 加载模型
    student = StudentDiscreteGesturesArchitecture()
    student.load_state_dict(
        torch.load('student_models/student_final.pt', map_location='cpu')
    )
    student.eval()

    # 准备输入（16通道EMG，2000样本 = 1秒@2000Hz）
    emg_data = torch.randn(1, 16, 2000)

    # 推理
    with torch.no_grad():
        logits = student(emg_data)
        probs = torch.sigmoid(logits)

    # logits shape: (1, 9, downsampled_time)
    # 9个手势的概率预测

    print(f"输出shape: {probs.shape}")
    print(f"各手势概率: {probs[0, :, 0]}")  # 第一个时间步

    # 手势类型对应关系
    gesture_names = [
        'index_press', 'index_release',
        'middle_press', 'middle_release',
        'thumb_click', 'thumb_down',
        'thumb_in', 'thumb_out', 'thumb_up'
    ]

    # 找到最可能的手势
    max_prob, max_idx = probs[0, :, 0].max(dim=0)
    print(f"最可能的手势: {gesture_names[max_idx]} (概率={max_prob:.2%})")


# ============================================================
# 示例5：对比Teacher和Student性能
# ============================================================

def compare_models():
    import torch
    import time
    from student_network import StudentDiscreteGesturesArchitecture
    from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture

    # 加载模型
    teacher = DiscreteGesturesArchitecture()
    teacher.load_state_dict(
        torch.load('../../../../../logs/best_discrete_gestures.pt', map_location='cpu')
    )
    teacher.eval()

    student = StudentDiscreteGesturesArchitecture()
    student.load_state_dict(
        torch.load('student_models/student_final.pt', map_location='cpu')
    )
    student.eval()

    # 测试数据
    test_input = torch.randn(1, 16, 2000)

    # Teacher推理
    start = time.time()
    with torch.no_grad():
        teacher_out = teacher(test_input)
    teacher_time = time.time() - start

    # Student推理
    start = time.time()
    with torch.no_grad():
        student_out = student(test_input)
    student_time = time.time() - start

    print(f"Teacher推理时间: {teacher_time*1000:.2f} ms")
    print(f"Student推理时间: {student_time*1000:.2f} ms")
    print(f"加速比: {teacher_time/student_time:.2f}x")

    # 输出相似度
    teacher_probs = torch.sigmoid(teacher_out)
    student_probs = torch.sigmoid(student_out)
    cosine_sim = torch.nn.functional.cosine_similarity(
        teacher_probs.flatten(),
        student_probs.flatten(),
        dim=0
    )
    print(f"输出相似度（余弦）: {cosine_sim:.4f}")


# ============================================================
# 示例6：导出模型为ONNX（便于部署）
# ============================================================

def export_onnx():
    import torch
    from student_network import StudentDiscreteGesturesArchitecture

    # 加载模型
    student = StudentDiscreteGesturesArchitecture()
    student.load_state_dict(
        torch.load('student_models/student_final.pt', map_location='cpu')
    )
    student.eval()

    # 导出ONNX
    dummy_input = torch.randn(1, 16, 2000)
    
    torch.onnx.export(
        student,
        dummy_input,
        "student_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['emg_input'],
        output_names=['gesture_logits'],
        dynamic_axes={
            'emg_input': {2: 'sequence_length'},
            'gesture_logits': {2: 'output_length'}
        }
    )
    
    print("✅ 模型已导出为: student_model.onnx")
    print("   可用于ONNX Runtime、TensorRT等推理引擎")


# ============================================================
# 常见问题解答
# ============================================================

"""
Q1: 训练不收敛怎么办？

A: 尝试以下方法：
   1. 降低学习率: --learning_rate 5e-4
   2. 减小temperature: --temperature 2.0
   3. 降低alpha: --alpha 0.3 (更依赖真实标签)
   4. 增大batch_size: --batch_size 32


Q2: Student准确率太低（比Teacher差很多）？

A: 可能的原因和解决方案：
   1. Student模型太小 → 增大模型（修改student_network.py）
   2. Alpha太低 → 提高到0.6-0.7
   3. Temperature太低 → 提高到4.0-5.0
   4. 训练不充分 → 增加max_epochs到150


Q3: 如何部署到边缘设备（如ESP32）？

A: 步骤：
   1. 导出ONNX: export_onnx()
   2. 用TensorFlow Lite转换器转为.tflite
   3. 量化为INT8（减小体积）
   4. 使用TFLite Micro在MCU上运行


Q4: 能否进一步压缩模型？

A: 可以尝试：
   1. 进一步减小conv_output_channels和lstm_hidden_size
   2. 使用剪枝（Pruning）
   3. 量化感知训练（QAT）
   4. 渐进式蒸馏（先蒸到中等模型，再蒸到小模型）


Q5: 如何监控训练？

A: 使用TensorBoard：
   tensorboard --logdir student_models/lightning_logs
   
   浏览器打开 http://localhost:6006
   可以看到：
   - train_loss / val_loss
   - train_distill_loss / train_task_loss
   - val_accuracy
   - 学习率曲线
"""
