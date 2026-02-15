"""
超轻量级Student模型 - 专为ESP32等微控制器设计

相比标准Student模型的进一步压缩方案：
1. TinyStudent: 1层LSTM，更少通道（~15万参数）
2. ConvOnlyStudent: 全卷积网络，无LSTM（~8万参数）
3. GRUStudent: 用GRU替代LSTM（~40万参数）

选择建议：
- ESP32 (520KB RAM): ConvOnlyStudent 或 TinyStudent + INT8量化
- 树莓派/手机: 标准Student模型
- 云端/PC: Teacher模型
"""

import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from generic_neuromotor_interface.networks import ReinhardCompression


class TinyStudentArchitecture(nn.Module):
    """
    超轻量级LSTM模型（~15万参数）
    
    压缩策略：
    - Conv通道: 128 -> 64 (2倍压缩)
    - LSTM层数: 2 -> 1 (单层)
    - LSTM隐藏: 256 -> 128 (2倍压缩)
    
    相比Teacher: ~2.3% (650万 -> 15万)
    相比Student: ~25% (60万 -> 15万)
    """
    
    def __init__(
        self,
        input_channels: int = 16,
        conv_output_channels: int = 64,   # Student: 128
        kernel_width: int = 21,
        stride: int = 10,
        lstm_hidden_size: int = 128,      # Student: 256
        lstm_num_layers: int = 1,         # Student: 2
        output_channels: int = 9,
    ):
        super().__init__()
        
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.left_context = kernel_width - 1
        self.stride = stride
        
        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)
        
        self.conv_layer = nn.Conv1d(
            input_channels, conv_output_channels,
            kernel_size=kernel_width, stride=stride
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.post_conv_layer_norm = nn.LayerNorm(normalized_shape=conv_output_channels)
        
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.0,  # 单层LSTM不需要dropout
        )
        
        self.post_lstm_layer_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
        self.projection = nn.Linear(lstm_hidden_size, output_channels)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.compression(inputs)
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x = self.post_conv_layer_norm(x)
        
        x, _ = self.lstm(x)
        x = self.post_lstm_layer_norm(x)
        
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x


class GRUStudentArchitecture(nn.Module):
    """
    GRU替代LSTM（~40万参数）
    
    GRU比LSTM参数少25%：
    - LSTM: 4个门（输入、遗忘、输出、cell）
    - GRU: 2个门（重置、更新）
    
    优势：参数少、速度快
    劣势：可能精度略低
    """
    
    def __init__(
        self,
        input_channels: int = 16,
        conv_output_channels: int = 128,
        kernel_width: int = 21,
        stride: int = 10,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        output_channels: int = 9,
    ):
        super().__init__()
        
        self.gru_num_layers = gru_num_layers
        self.gru_hidden_size = gru_hidden_size
        self.left_context = kernel_width - 1
        self.stride = stride
        
        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)
        
        self.conv_layer = nn.Conv1d(
            input_channels, conv_output_channels,
            kernel_size=kernel_width, stride=stride
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.post_conv_layer_norm = nn.LayerNorm(normalized_shape=conv_output_channels)
        
        # GRU替代LSTM
        self.gru = nn.GRU(
            input_size=conv_output_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=0.1 if gru_num_layers > 1 else 0.0,
        )
        
        self.post_gru_layer_norm = nn.LayerNorm(normalized_shape=gru_hidden_size)
        self.projection = nn.Linear(gru_hidden_size, output_channels)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.compression(inputs)
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x = self.post_conv_layer_norm(x)
        
        x, _ = self.gru(x)
        x = self.post_gru_layer_norm(x)
        
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x


class ConvOnlyStudentArchitecture(nn.Module):
    """
    全卷积网络（~8万参数）- 最适合ESP32
    
    完全移除LSTM，用多层卷积替代：
    - Conv1: 16 -> 64 通道
    - Conv2: 64 -> 64 通道（扩大感受野）
    - Conv3: 64 -> 32 通道
    - Linear: 32 -> 9
    
    优势：
    - 参数最少（~8万）
    - 推理速度最快（无循环依赖）
    - 内存占用最小
    - 易于量化和部署
    
    劣势：
    - 无法建模长期依赖
    - 可能精度最低
    """
    
    def __init__(
        self,
        input_channels: int = 16,
        hidden_channels: int = 64,
        output_channels: int = 9,
    ):
        super().__init__()
        
        self.left_context = 20  # Conv1(20) = 20
        self.stride = 10
        
        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)
        
        # 三层卷积网络
        self.conv1 = nn.Conv1d(
            input_channels, hidden_channels,
            kernel_size=21, stride=10
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        # 第二层：扩大时间感受野
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels,
            kernel_size=5, stride=1, padding=2
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        
        # 第三层：降维
        self.conv3 = nn.Conv1d(
            hidden_channels, 32,
            kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        
        # 输出层
        self.projection = nn.Conv1d(32, output_channels, kernel_size=1)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.compression(inputs)
        
        # Conv块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Conv块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Conv块3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # 输出
        x = self.projection(x)
        return x


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
    }


if __name__ == "__main__":
    print("="*70)
    print("超轻量级模型对比 - ESP32部署方案")
    print("="*70)
    
    # 创建所有模型
    models = {
        "TinyStudent (1-LSTM)": TinyStudentArchitecture(),
        "GRUStudent (2-GRU)": GRUStudentArchitecture(),
        "ConvOnly (No-RNN)": ConvOnlyStudentArchitecture(),
    }
    
    # 参考模型
    from student_network import StudentDiscreteGesturesArchitecture
    from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture
    
    teacher = DiscreteGesturesArchitecture()
    student = StudentDiscreteGesturesArchitecture()
    
    print(f"\n📊 模型参数对比:\n")
    print(f"{'模型':<25} {'参数量':>12} {'相对Teacher':>12} {'相对Student':>12}")
    print("-" * 70)
    
    teacher_params = count_parameters(teacher)['total']
    student_params = count_parameters(student)['total']
    
    print(f"{'Teacher (Meta大模型)':<25} {teacher_params:>12,} {'100.0%':>12} {'-':>12}")
    print(f"{'Student (标准小模型)':<25} {student_params:>12,} {f'{student_params/teacher_params:.1%}':>12} {'100.0%':>12}")
    print("-" * 70)
    
    for name, model in models.items():
        params = count_parameters(model)['total']
        print(f"{name:<25} {params:>12,} {f'{params/teacher_params:.1%}':>12} {f'{params/student_params:.1%}':>12}")
    
    # 测试推理
    print(f"\n\n🧪 推理测试（输入: 1×16×2000）:\n")
    dummy_input = torch.randn(1, 16, 2000)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"{name:<25} 输出: {output.shape}")
    
    # ESP32部署建议
    print(f"\n\n💡 ESP32部署建议:\n")
    
    conv_params = count_parameters(models["ConvOnly (No-RNN)"])['total']
    tiny_params = count_parameters(models["TinyStudent (1-LSTM)"])['total']
    
    # FP32大小
    conv_size_fp32 = conv_params * 4 / 1024  # KB
    tiny_size_fp32 = tiny_params * 4 / 1024
    
    # INT8大小
    conv_size_int8 = conv_params * 1 / 1024
    tiny_size_int8 = tiny_params * 1 / 1024
    
    print(f"1️⃣  ConvOnly模型:")
    print(f"   - FP32: {conv_size_fp32:.1f} KB")
    print(f"   - INT8量化后: {conv_size_int8:.1f} KB")
    print(f"   - 推荐设备: ESP32 (520KB RAM)")
    print(f"   - 特点: 最快、最小、无状态")
    
    print(f"\n2️⃣  TinyStudent模型:")
    print(f"   - FP32: {tiny_size_fp32:.1f} KB")
    print(f"   - INT8量化后: {tiny_size_int8:.1f} KB")
    print(f"   - 推荐设备: ESP32-S3 (更大RAM) 或树莓派Pico")
    print(f"   - 特点: 中等大小、有状态、精度较好")
    
    print(f"\n3️⃣  GRUStudent模型:")
    print(f"   - 推荐设备: 树莓派、手机")
    print(f"   - 特点: ESP32可能内存不足")
    
    print(f"\n\n🎯 推荐方案:")
    print(f"   ESP32: ConvOnly + INT8量化 (~{conv_size_int8:.0f}KB)")
    print(f"   ESP32-S3: TinyStudent + INT8量化 (~{tiny_size_int8:.0f}KB)")
    print(f"   树莓派/手机: 标准Student模型")
