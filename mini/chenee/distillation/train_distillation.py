#!/usr/bin/env python3
"""
知识蒸馏训练脚本

使用预训练的Teacher模型（Meta大模型）蒸馏到轻量级Student模型

使用方法:
    python train_distillation.py --teacher_checkpoint ../logs/best_discrete_gestures.pt
"""

import argparse
from pathlib import Path
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture
from generic_neuromotor_interface.data_module import EmgDataModule
from student_network import StudentDiscreteGesturesArchitecture
from distillation_module import DistillationModule


def load_teacher_model(checkpoint_path: str) -> DiscreteGesturesArchitecture:
    """
    加载预训练的Teacher模型
    
    Parameters
    ----------
    checkpoint_path : str
        Teacher模型权重路径（.pt或.ckpt文件）
        
    Returns
    -------
    teacher : DiscreteGesturesArchitecture
        加载权重后的Teacher模型
    """
    print(f"\n📂 加载Teacher模型: {checkpoint_path}")
    
    # 创建Teacher网络
    teacher = DiscreteGesturesArchitecture()
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理不同的checkpoint格式
    if 'state_dict' in checkpoint:
        # Lightning checkpoint格式
        state_dict = checkpoint['state_dict']
        # 移除'network.'前缀
        state_dict = {k.replace('network.', ''): v for k, v in state_dict.items() 
                     if k.startswith('network.')}
    else:
        # 直接的state_dict
        state_dict = checkpoint
    
    teacher.load_state_dict(state_dict)
    teacher.eval()
    
    print(f"✅ Teacher模型加载成功")
    print(f"   参数量: {sum(p.numel() for p in teacher.parameters()):,}")
    
    return teacher


def create_student_model() -> StudentDiscreteGesturesArchitecture:
    """创建Student模型"""
    print("\n🎓 创建Student模型")
    
    student = StudentDiscreteGesturesArchitecture(
        input_channels=16,
        conv_output_channels=128,  # Teacher: 512
        kernel_width=21,
        stride=10,
        lstm_hidden_size=256,      # Teacher: 512
        lstm_num_layers=2,         # Teacher: 3
        output_channels=9,
    )
    
    params = student.count_parameters()
    print(f"✅ Student模型创建成功")
    print(f"   参数量: {params['total']:,}")
    print(f"   - Conv: {params['conv']:,}")
    print(f"   - LSTM: {params['lstm']:,}")
    print(f"   - 投影: {params['projection']:,}")
    
    return student


def setup_data_module(
    data_dir: str,
    split_csv: str,
    batch_size: int = 16,
    window_duration: float = 0.25,
    window_stride: int = 40,
) -> EmgDataModule:
    """
    设置数据模块
    
    Parameters
    ----------
    data_dir : str
        数据目录路径
    split_csv : str
        数据划分CSV路径
    batch_size : int
        批次大小
    window_duration : float
        窗口时长（秒）
    window_stride : int
        窗口步长（样本数）
        
    Returns
    -------
    data_module : EmgDataModule
        配置好的数据模块
    """
    print(f"\n📊 设置数据模块")
    print(f"   数据目录: {data_dir}")
    print(f"   划分文件: {split_csv}")
    print(f"   批次大小: {batch_size}")
    
    data_module = EmgDataModule(
        task="discrete_gestures",
        data_dir=data_dir,
        split_csv=split_csv,
        batch_size=batch_size,
        window_duration=window_duration,
        window_stride=window_stride,
        num_workers=4,
    )
    
    return data_module


def train_distillation(
    teacher_checkpoint: str,
    data_dir: str,
    split_csv: str,
    output_dir: str = "./distillation_output",
    batch_size: int = 16,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    temperature: float = 3.0,
    alpha: float = 0.5,
    gpus: int = 1,
):
    """
    执行知识蒸馏训练
    
    Parameters
    ----------
    teacher_checkpoint : str
        Teacher模型检查点路径
    data_dir : str
        数据目录
    split_csv : str
        数据划分CSV
    output_dir : str
        输出目录
    batch_size : int
        批次大小
    max_epochs : int
        最大训练轮数
    learning_rate : float
        学习率
    temperature : float
        蒸馏温度（推荐2-5）
    alpha : float
        蒸馏损失权重（0-1）
    gpus : int
        GPU数量
    """
    
    print("\n" + "="*60)
    print("开始知识蒸馏训练")
    print("="*60)
    
    # 1. 加载Teacher模型
    teacher = load_teacher_model(teacher_checkpoint)
    
    # 2. 创建Student模型
    student = create_student_model()
    
    # 比较参数量
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"\n📊 模型对比:")
    print(f"   Teacher: {teacher_params:,} 参数")
    print(f"   Student: {student_params:,} 参数")
    print(f"   压缩比: {student_params/teacher_params:.1%}")
    
    # 3. 设置数据模块
    data_module = setup_data_module(
        data_dir=data_dir,
        split_csv=split_csv,
        batch_size=batch_size,
    )
    
    # 4. 创建蒸馏模块
    print(f"\n🔥 创建蒸馏训练模块")
    print(f"   温度: {temperature}")
    print(f"   Alpha: {alpha} (distill={alpha}, task={1-alpha})")
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    
    distill_module = DistillationModule(
        student_network=student,
        teacher_network=teacher,
        optimizer=optimizer,
        learning_rate=learning_rate,
        lr_scheduler_milestones=[60, 80],
        lr_scheduler_factor=0.1,
        warmup_start_factor=0.1,
        warmup_end_factor=1.0,
        warmup_total_epochs=5,
        gradient_clip_val=1.0,
        temperature=temperature,
        alpha=alpha,
    )
    
    # 5. 设置Callbacks
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename='student-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 6. 创建Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=False,
    )
    
    # 7. 开始训练
    print(f"\n🚀 开始训练（最多{max_epochs}轮）")
    print(f"   输出目录: {output_path}")
    
    trainer.fit(distill_module, datamodule=data_module)
    
    # 8. 保存最终模型
    final_model_path = output_path / "student_final.pt"
    torch.save(student.state_dict(), final_model_path)
    
    print(f"\n✅ 训练完成！")
    print(f"   最佳模型: {checkpoint_callback.best_model_path}")
    print(f"   最终模型: {final_model_path}")
    
    return distill_module, trainer


def main():
    parser = argparse.ArgumentParser(description="知识蒸馏训练脚本")
    
    # 必需参数
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="Teacher模型检查点路径",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="EMG数据目录",
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        required=True,
        help="数据划分CSV文件路径",
    )
    
    # 可选参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distillation_output",
        help="输出目录（默认: ./distillation_output）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批次大小（默认: 16）",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="最大训练轮数（默认: 100）",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="学习率（默认: 1e-3）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="蒸馏温度（默认: 3.0，推荐2-5）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="蒸馏损失权重（默认: 0.5，范围0-1）",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPU数量（默认: 1）",
    )
    
    args = parser.parse_args()
    
    # 执行训练
    train_distillation(
        teacher_checkpoint=args.teacher_checkpoint,
        data_dir=args.data_dir,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        alpha=args.alpha,
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
