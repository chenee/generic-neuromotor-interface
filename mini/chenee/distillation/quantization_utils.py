"""
模型量化工具 - 将FP32模型转为INT8

INT8量化可以：
1. 模型大小缩小4倍
2. 推理速度提升2-4倍
3. 内存占用减少75%

适用于ESP32等资源受限设备
"""

import torch
import torch.quantization as quantization
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))


def quantize_model_dynamic(
    model: torch.nn.Module,
    output_path: str,
    qconfig_spec: set = None
) -> torch.nn.Module:
    """
    动态量化（推荐用于LSTM/GRU模型）
    
    只量化权重，激活值在运行时动态量化
    
    Parameters
    ----------
    model : torch.nn.Module
        待量化的FP32模型
    output_path : str
        量化后模型保存路径
    qconfig_spec : set
        要量化的层类型，默认{nn.Linear, nn.LSTM, nn.GRU}
    
    Returns
    -------
    quantized_model : torch.nn.Module
        量化后的模型
    """
    if qconfig_spec is None:
        qconfig_spec = {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}
    
    model.eval()
    
    # 动态量化
    quantized_model = quantization.quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=torch.qint8
    )
    
    # 保存
    torch.save(quantized_model.state_dict(), output_path)
    
    print(f"✅ 动态量化完成: {output_path}")
    
    return quantized_model


def quantize_model_static(
    model: torch.nn.Module,
    calibration_data: torch.Tensor,
    output_path: str
) -> torch.nn.Module:
    """
    静态量化（推荐用于纯卷积模型）
    
    权重和激活值都预先量化，精度最高
    
    Parameters
    ----------
    model : torch.nn.Module
        待量化的FP32模型
    calibration_data : torch.Tensor
        校准数据，用于统计激活值范围
    output_path : str
        量化后模型保存路径
    
    Returns
    -------
    quantized_model : torch.nn.Module
        量化后的模型
    """
    model.eval()
    
    # 设置量化配置
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    
    # 准备量化
    quantization.prepare(model, inplace=True)
    
    # 校准（用真实数据运行一次）
    with torch.no_grad():
        _ = model(calibration_data)
    
    # 转换为量化模型
    quantization.convert(model, inplace=True)
    
    # 保存
    torch.save(model.state_dict(), output_path)
    
    print(f"✅ 静态量化完成: {output_path}")
    
    return model


def compare_model_size(
    fp32_model: torch.nn.Module,
    quantized_model: torch.nn.Module
) -> dict:
    """对比量化前后的模型大小"""
    
    def get_model_size(model):
        torch.save(model.state_dict(), "/tmp/temp_model.pt")
        size = Path("/tmp/temp_model.pt").stat().st_size
        Path("/tmp/temp_model.pt").unlink()
        return size
    
    fp32_size = get_model_size(fp32_model)
    quantized_size = get_model_size(quantized_model)
    
    return {
        "fp32_size_kb": fp32_size / 1024,
        "quantized_size_kb": quantized_size / 1024,
        "compression_ratio": fp32_size / quantized_size,
    }


def export_to_onnx_int8(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 16, 2000)
):
    """
    导出为ONNX格式（INT8）
    
    ONNX可以被多种推理引擎使用：
    - ONNX Runtime
    - TensorRT
    - TFLite (需要进一步转换)
    """
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['emg_input'],
        output_names=['gesture_logits'],
        dynamic_axes={
            'emg_input': {2: 'sequence_length'},
            'gesture_logits': {2: 'output_length'}
        }
    )
    
    print(f"✅ ONNX导出完成: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("模型量化示例")
    print("="*60)
    
    # 1. 导入模型
    from ultra_light_models import (
        TinyStudentArchitecture,
        ConvOnlyStudentArchitecture,
        count_parameters
    )
    
    # 2. 创建模型
    print("\n创建ConvOnly模型（最适合量化）...")
    model = ConvOnlyStudentArchitecture()
    model.eval()
    
    # 3. 动态量化
    print("\n执行动态量化...")
    quantized_model = quantize_model_dynamic(
        model,
        output_path="convonly_int8.pt",
        qconfig_spec={torch.nn.Conv1d, torch.nn.Linear}
    )
    
    # 4. 对比大小
    print("\n📊 模型大小对比:")
    size_info = compare_model_size(model, quantized_model)
    
    print(f"   FP32模型: {size_info['fp32_size_kb']:.1f} KB")
    print(f"   INT8模型: {size_info['quantized_size_kb']:.1f} KB")
    print(f"   压缩比: {size_info['compression_ratio']:.2f}x")
    
    # 5. 测试精度损失
    print("\n🧪 精度测试:")
    dummy_input = torch.randn(1, 16, 2000)
    
    with torch.no_grad():
        fp32_output = model(dummy_input)
        int8_output = quantized_model(dummy_input)
    
    # 计算输出差异
    mae = torch.mean(torch.abs(fp32_output - int8_output)).item()
    max_diff = torch.max(torch.abs(fp32_output - int8_output)).item()
    
    print(f"   平均绝对误差: {mae:.6f}")
    print(f"   最大误差: {max_diff:.6f}")
    print(f"   相对误差: {mae/torch.mean(torch.abs(fp32_output)).item():.2%}")
    
    # 6. 导出ONNX
    print("\n📦 导出ONNX格式...")
    export_to_onnx_int8(quantized_model, "convonly_int8.onnx")
    
    print("\n✨ 量化完成！")
    print("\n下一步:")
    print("   1. 用ONNX转换为TFLite: onnx2tf convonly_int8.onnx")
    print("   2. 或直接在PyTorch Mobile上运行")
    print("   3. ESP32部署需要转为TFLite Micro格式")
