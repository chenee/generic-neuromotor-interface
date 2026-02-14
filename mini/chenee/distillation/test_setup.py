#!/usr/bin/env python3
"""
快速验证脚本

测试Student模型和蒸馏模块是否能正常工作
不需要真实数据，使用随机数据验证
"""

import torch
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from student_network import StudentDiscreteGesturesArchitecture
from distillation_module import DistillationLoss
from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture


def test_student_network():
    """测试Student网络"""
    print("\n" + "="*60)
    print("1️⃣  测试Student网络")
    print("="*60)
    
    # 创建模型
    student = StudentDiscreteGesturesArchitecture()
    
    # 统计参数
    params = student.count_parameters()
    print(f"\n参数统计:")
    print(f"  总参数: {params['total']:,}")
    print(f"  Conv层: {params['conv']:,}")
    print(f"  LSTM层: {params['lstm']:,}")
    print(f"  投影层: {params['projection']:,}")
    
    # 测试前向传播
    batch_size = 2
    num_channels = 16
    seq_length = 1000
    
    dummy_input = torch.randn(batch_size, num_channels, seq_length)
    
    print(f"\n前向传播测试:")
    print(f"  输入shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = student(dummy_input)
    
    output_length = len(torch.arange(seq_length)[student.left_context::student.stride])
    expected_shape = (batch_size, 9, output_length)
    
    print(f"  输出shape: {output.shape}")
    print(f"  预期shape: {expected_shape}")
    
    assert output.shape == expected_shape, f"Shape不匹配！"
    print("  ✅ 测试通过")
    
    return student


def test_teacher_student_comparison():
    """对比Teacher和Student"""
    print("\n" + "="*60)
    print("2️⃣  对比Teacher和Student模型")
    print("="*60)
    
    # 创建两个模型
    teacher = DiscreteGesturesArchitecture()
    student = StudentDiscreteGesturesArchitecture()
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"\n模型对比:")
    print(f"  Teacher参数: {teacher_params:,}")
    print(f"  Student参数: {student_params:,}")
    print(f"  压缩比: {student_params/teacher_params:.1%}")
    print(f"  参数减少: {teacher_params - student_params:,}")
    
    # 测试推理速度（CPU）
    import time
    
    dummy_input = torch.randn(1, 16, 2000)
    
    # Teacher
    teacher.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = teacher(dummy_input)
    teacher_time = (time.time() - start) / 100
    
    # Student
    student.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = student(dummy_input)
    student_time = (time.time() - start) / 100
    
    print(f"\n推理速度（CPU，100次平均）:")
    print(f"  Teacher: {teacher_time*1000:.2f} ms")
    print(f"  Student: {student_time*1000:.2f} ms")
    print(f"  加速比: {teacher_time/student_time:.2f}x")
    
    return teacher, student


def test_distillation_loss():
    """测试蒸馏损失"""
    print("\n" + "="*60)
    print("3️⃣  测试蒸馏损失函数")
    print("="*60)
    
    # 创建损失函数
    distill_loss = DistillationLoss(temperature=3.0, alpha=0.5)
    
    print(f"\n配置:")
    print(f"  Temperature: {distill_loss.temperature}")
    print(f"  Alpha: {distill_loss.alpha}")
    
    # 模拟数据
    B, C, T = 4, 9, 100
    
    student_logits = torch.randn(B, C, T)
    teacher_logits = torch.randn(B, C, T)
    targets = torch.randint(0, 2, (B, C, T)).float()
    mask = torch.ones(B, C, T)
    
    # 计算损失
    loss_dict = distill_loss(student_logits, teacher_logits, targets, mask)
    
    print(f"\n损失值:")
    print(f"  Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"  Distill Loss: {loss_dict['distill_loss']:.4f}")
    print(f"  Task Loss: {loss_dict['task_loss']:.4f}")
    
    # 验证总损失计算
    expected_total = (
        distill_loss.alpha * loss_dict['distill_loss'] + 
        (1 - distill_loss.alpha) * loss_dict['task_loss']
    )
    
    assert torch.allclose(loss_dict['total_loss'], expected_total), "总损失计算错误！"
    print("  ✅ 损失计算正确")


def test_output_consistency():
    """测试Teacher和Student输出一致性"""
    print("\n" + "="*60)
    print("4️⃣  测试输出一致性")
    print("="*60)
    
    teacher = DiscreteGesturesArchitecture()
    student = StudentDiscreteGesturesArchitecture()
    
    # 相同输入
    dummy_input = torch.randn(1, 16, 2000)
    
    teacher.eval()
    student.eval()
    
    with torch.no_grad():
        teacher_out = teacher(dummy_input)
        student_out = student(dummy_input)
    
    print(f"\n输出shape:")
    print(f"  Teacher: {teacher_out.shape}")
    print(f"  Student: {student_out.shape}")
    
    # 验证shape一致
    assert teacher_out.shape == student_out.shape, "输出shape不一致！"
    print("  ✅ Shape一致")
    
    # 检查值分布（未训练，应该不同）
    teacher_mean = teacher_out.mean().item()
    student_mean = student_out.mean().item()
    
    print(f"\n输出统计（未训练状态）:")
    print(f"  Teacher均值: {teacher_mean:.4f}")
    print(f"  Student均值: {student_mean:.4f}")
    print(f"  Teacher标准差: {teacher_out.std().item():.4f}")
    print(f"  Student标准差: {student_out.std().item():.4f}")


def main():
    print("\n" + "="*60)
    print("🧪 知识蒸馏模块验证")
    print("="*60)
    
    try:
        # 测试1：Student网络
        student = test_student_network()
        
        # 测试2：Teacher vs Student对比
        teacher, student = test_teacher_student_comparison()
        
        # 测试3：蒸馏损失
        test_distillation_loss()
        
        # 测试4：输出一致性
        test_output_consistency()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！可以开始训练了")
        print("="*60)
        
        print("\n📖 下一步:")
        print("   1. 准备数据集和Teacher模型")
        print("   2. 运行: python train_distillation.py --teacher_checkpoint <path> \\")
        print("           --data_dir <data> --split_csv <split>")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
