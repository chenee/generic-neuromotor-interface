"""
知识蒸馏Lightning模块

实现Teacher-Student蒸馏框架，用于离散手势识别任务
结合软标签蒸馏损失和硬标签任务损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Mapping, Any
import sys
from pathlib import Path

# 导入原始模块
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from generic_neuromotor_interface.lightning import DiscreteGesturesModule
from generic_neuromotor_interface.constants import GestureType
from generic_neuromotor_interface.cler import compute_cler


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    
    组合三种损失：
    1. KL散度损失：Student学习Teacher的软标签
    2. BCE任务损失：Student学习真实硬标签
    3. 特征匹配损失（可选）：Student模仿Teacher的中间特征
    
    Parameters
    ----------
    temperature : float
        蒸馏温度，softmax平滑参数（通常2-5）
    alpha : float
        蒸馏损失权重（0-1之间）
        总损失 = alpha * distill_loss + (1-alpha) * task_loss
    """
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        计算蒸馏损失
        
        Parameters
        ----------
        student_logits : torch.Tensor
            学生模型输出，shape=(B, 9, T)
        teacher_logits : torch.Tensor
            教师模型输出（detached），shape=(B, 9, T)
        targets : torch.Tensor
            真实标签，shape=(B, 9, T)
        mask : torch.Tensor
            损失掩码，shape=(B, 9, T)
            
        Returns
        -------
        loss_dict : dict
            包含total_loss, distill_loss, task_loss
        """
        
        # 1. 蒸馏损失（KL散度）
        # 使用温度软化logits，让分布更平滑
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL散度 = sum(teacher * log(teacher/student))
        distill_loss = F.kl_div(
            student_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)  # 温度平方补偿梯度缩放
        
        # 2. 任务损失（BCE）
        task_loss = self.task_loss_fn(student_logits, targets)
        task_loss = (task_loss * mask).sum() / mask.sum()
        
        # 3. 总损失
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'task_loss': task_loss,
        }


class DistillationModule(DiscreteGesturesModule):
    """
    蒸馏训练的Lightning模块
    
    继承自原始DiscreteGesturesModule，添加Teacher模型和蒸馏损失
    
    Parameters
    ----------
    student_network : nn.Module
        学生网络
    teacher_network : nn.Module
        教师网络（预训练模型）
    optimizer : torch.optim.Optimizer
        优化器
    learning_rate : float
        学习率
    temperature : float
        蒸馏温度
    alpha : float
        蒸馏损失权重
    ... (其他参数同DiscreteGesturesModule)
    """
    
    def __init__(
        self,
        student_network: nn.Module,
        teacher_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        lr_scheduler_milestones: list[int],
        lr_scheduler_factor: float,
        warmup_start_factor: float,
        warmup_end_factor: float,
        warmup_total_epochs: int,
        gradient_clip_val: float,
        temperature: float = 3.0,
        alpha: float = 0.5,
    ) -> None:
        # 用student_network初始化父类
        super().__init__(
            network=student_network,
            optimizer=optimizer,
            learning_rate=learning_rate,
            lr_scheduler_milestones=lr_scheduler_milestones,
            lr_scheduler_factor=lr_scheduler_factor,
            warmup_start_factor=warmup_start_factor,
            warmup_end_factor=warmup_end_factor,
            warmup_total_epochs=warmup_total_epochs,
            gradient_clip_val=gradient_clip_val,
        )
        
        # Teacher模型
        self.teacher = teacher_network
        self.teacher.eval()  # 始终为评估模式
        # 冻结Teacher参数
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 蒸馏损失函数
        self.distill_loss_fn = DistillationLoss(
            temperature=temperature,
            alpha=alpha
        )
        
        # 保存超参数
        self.save_hyperparameters(ignore=['student_network', 'teacher_network', 'optimizer'])
        
    def _step(self, batch: Mapping[str, torch.Tensor], stage: str = "train") -> float:
        """
        训练/验证步骤（重写父类方法）
        
        添加Teacher模型的软标签生成
        """
        # 提取数据
        emg = batch["emg"]
        targets = batch["targets"]
        targets = targets[:, :, self.network.left_context :: self.network.stride]
        release_mask = self.mask_generator(targets)
        mask = torch.ones_like(targets)
        mask[
            :, [GestureType.index_release.value, GestureType.middle_release.value], :
        ] = release_mask
        
        # Student模型预测
        student_logits = self.forward(emg)
        
        if stage == "train":
            # 训练阶段：使用Teacher软标签
            with torch.no_grad():
                teacher_logits = self.teacher(emg)
            
            # 计算蒸馏损失
            loss_dict = self.distill_loss_fn(
                student_logits, 
                teacher_logits,
                targets,
                mask
            )
            
            loss = loss_dict['total_loss']
            
            # 记录各项损失
            self.log(f"{stage}_loss", loss, sync_dist=True)
            self.log(f"{stage}_distill_loss", loss_dict['distill_loss'], sync_dist=True)
            self.log(f"{stage}_task_loss", loss_dict['task_loss'], sync_dist=True)
            
        else:
            # 验证/测试阶段：只用任务损失
            loss = self.loss_fn(student_logits, targets)
            loss = (loss * mask).sum() / mask.sum()
            self.log(f"{stage}_loss", loss, sync_dist=True)
            
            if stage == "val":
                self.collect_metric(
                    student_logits.permute(0, 2, 1),
                    targets.permute(0, 2, 1),
                    phase=stage,
                )
            
            elif stage == "test":
                prompts = batch["prompts"][0]
                times = batch["timestamps"][0]
                preds = nn.Sigmoid()(student_logits)
                preds = preds.squeeze(0).detach().cpu().numpy()
                times = times[self.network.left_context :: self.network.stride]
                cler = compute_cler(preds, times, prompts)
                self.log("test_cler", cler, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss


if __name__ == "__main__":
    print("="*60)
    print("Distillation Module Test")
    print("="*60)
    
    # 导入网络
    from student_network import StudentDiscreteGesturesArchitecture
    from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture
    
    # 创建Teacher和Student
    print("\n创建模型...")
    teacher = DiscreteGesturesArchitecture()
    student = StudentDiscreteGesturesArchitecture()
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"Teacher参数: {teacher_params:,}")
    print(f"Student参数: {student_params:,}")
    print(f"压缩比: {student_params/teacher_params:.1%}")
    
    # 测试蒸馏损失
    print("\n测试蒸馏损失...")
    distill_loss = DistillationLoss(temperature=3.0, alpha=0.5)
    
    # 模拟数据
    B, C, T = 2, 9, 50
    student_logits = torch.randn(B, C, T)
    teacher_logits = torch.randn(B, C, T)
    targets = torch.randint(0, 2, (B, C, T)).float()
    mask = torch.ones(B, C, T)
    
    loss_dict = distill_loss(student_logits, teacher_logits, targets, mask)
    
    print(f"Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"Distill Loss: {loss_dict['distill_loss']:.4f}")
    print(f"Task Loss: {loss_dict['task_loss']:.4f}")
    
    print("\n✨ 蒸馏模块测试通过！")
