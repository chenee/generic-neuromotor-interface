"""
知识蒸馏模块

用于将Meta的离散手势识别大模型（650万参数）蒸馏到轻量级Student模型（60万参数）
"""

__version__ = "0.1.0"

from .student_network import StudentDiscreteGesturesArchitecture
from .distillation_module import DistillationModule, DistillationLoss

__all__ = [
    "StudentDiscreteGesturesArchitecture",
    "DistillationModule",
    "DistillationLoss",
]
