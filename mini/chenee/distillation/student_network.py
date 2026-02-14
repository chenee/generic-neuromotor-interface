"""
学生网络：离散手势识别的轻量级模型

相比Teacher模型（650万参数），Student模型约60万参数（~10%）
适合部署到资源受限的设备（如微控制器、移动端）
"""

import torch
from torch import nn
import sys
from pathlib import Path

# 导入Teacher模型的压缩层
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from generic_neuromotor_interface.networks import ReinhardCompression


class StudentDiscreteGesturesArchitecture(nn.Module):
    """
    轻量级离散手势识别网络
    
    与Teacher模型对比：
    - Conv通道数：512 -> 128 (4倍压缩)
    - LSTM层数：3 -> 2 (减少1层)
    - LSTM隐藏单元：512 -> 256 (2倍压缩)
    - 总参数量：~650万 -> ~60万（约10%）
    
    Parameters
    ----------
    input_channels : int
        输入EMG通道数（默认16）
    conv_output_channels : int
        卷积输出通道数（默认128，Teacher为512）
    kernel_width : int
        卷积核宽度（保持21）
    stride : int
        卷积步长（保持10）
    lstm_hidden_size : int
        LSTM隐藏单元数（默认256，Teacher为512）
    lstm_num_layers : int
        LSTM层数（默认2，Teacher为3）
    output_channels : int
        输出手势类别数（9种手势）
    """

    def __init__(
        self,
        input_channels: int = 16,
        conv_output_channels: int = 128,  # Teacher: 512
        kernel_width: int = 21,
        stride: int = 10,
        lstm_hidden_size: int = 256,      # Teacher: 512
        lstm_num_layers: int = 2,         # Teacher: 3
        output_channels: int = 9,
    ) -> None:
        super().__init__()

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.left_context = kernel_width - 1
        self.stride = stride

        # ============ 与Teacher相同的层 ============
        # Reinhard动态范围压缩
        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)

        # Conv1d层（通道数减少）
        self.conv_layer = nn.Conv1d(
            input_channels,
            conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
        )

        # ReLU激活
        self.relu = nn.ReLU()

        # Dropout（保持0.1）
        self.dropout = nn.Dropout(p=0.1)

        # LayerNorm
        self.post_conv_layer_norm = nn.LayerNorm(normalized_shape=conv_output_channels)

        # ============ 精简后的LSTM ============
        # LSTM层数减少到2层，隐藏单元减少到256
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1 if lstm_num_layers > 1 else 0.0,  # 单层LSTM不用dropout
        )

        # LayerNorm
        self.post_lstm_layer_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)

        # 输出投影层
        self.projection = nn.Linear(lstm_hidden_size, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播（与Teacher结构完全一致）

        Parameters
        ----------
        inputs : torch.Tensor
            输入EMG数据，shape=(batch_size, 16, sequence_length)

        Returns
        -------
        output : torch.Tensor
            手势预测logits，shape=(batch_size, 9, downsampled_length)
        """

        # Reinhard压缩
        x = self.compression(inputs)

        # 卷积层
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.dropout(x)

        # LayerNorm (需要转置)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.post_conv_layer_norm(x)

        # LSTM
        x, _ = self.lstm(x)

        # LayerNorm
        x = self.post_lstm_layer_norm(x)

        # 输出投影
        x = self.projection(x)
        x = x.permute(0, 2, 1)  # (B, T, 9) -> (B, 9, T)

        return x

    def count_parameters(self) -> dict:
        """统计模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 分层统计
        conv_params = sum(p.numel() for p in self.conv_layer.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        proj_params = sum(p.numel() for p in self.projection.parameters())
        
        return {
            "total": total,
            "trainable": trainable,
            "conv": conv_params,
            "lstm": lstm_params,
            "projection": proj_params,
        }


if __name__ == "__main__":
    # 测试学生网络
    print("="*60)
    print("Student Network Architecture")
    print("="*60)
    
    # 创建模型
    student = StudentDiscreteGesturesArchitecture()
    
    # 统计参数
    params = student.count_parameters()
    print(f"\n📊 参数统计:")
    print(f"  总参数量: {params['total']:,}")
    print(f"  可训练参数: {params['trainable']:,}")
    print(f"  Conv层: {params['conv']:,}")
    print(f"  LSTM层: {params['lstm']:,}")
    print(f"  投影层: {params['projection']:,}")
    
    # 测试前向传播
    batch_size = 2
    num_channels = 16
    seq_length = 1000
    
    dummy_input = torch.randn(batch_size, num_channels, seq_length)
    output = student(dummy_input)
    
    output_length = len(torch.arange(seq_length)[student.left_context::student.stride])
    
    print(f"\n✅ 前向传播测试:")
    print(f"  输入shape: {dummy_input.shape}")
    print(f"  输出shape: {output.shape}")
    print(f"  预期输出shape: ({batch_size}, 9, {output_length})")
    
    assert output.shape == (batch_size, 9, output_length), "输出shape不匹配！"
    print("\n✨ Student模型测试通过！")
