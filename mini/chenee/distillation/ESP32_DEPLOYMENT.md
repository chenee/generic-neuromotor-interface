# ESP32部署指南 - 超轻量级手势识别

针对ESP32（~520KB RAM）的模型部署完整方案

## 🎯 模型选择

### 方案对比

| 模型 | 参数量 | FP32大小 | INT8大小 | 推荐设备 |
|------|--------|----------|----------|---------|
| **ConvOnly** | ~8万 | 320KB | **80KB** | ✅ ESP32 |
| **TinyStudent** | ~15万 | 600KB | **150KB** | ESP32-S3 |
| **GRUStudent** | ~40万 | 1.6MB | 400KB | 树莓派 |
| Standard Student | ~60万 | 2.4MB | 600KB | 手机/PC |

### ESP32推荐：ConvOnly + INT8

**为什么选择ConvOnly？**
- ✅ 最小模型（INT8仅80KB）
- ✅ 无LSTM，无状态，内存占用低
- ✅ 全卷积，推理速度快
- ✅ 易于量化，精度损失小

## 📋 完整部署流程

### Step 1: 训练超轻量级模型

```bash
cd distillation

# 1. 先验证模型结构
python3 ultra_light_models.py

# 2. 修改train_distillation.py，替换Student模型
# 将 StudentDiscreteGesturesArchitecture 改为 ConvOnlyStudentArchitecture
```

**修改训练脚本**：

```python
# 在 train_distillation.py 中
from ultra_light_models import ConvOnlyStudentArchitecture  # 新增

def create_student_model():
    # 使用ConvOnly模型
    student = ConvOnlyStudentArchitecture(
        input_channels=16,
        hidden_channels=64,  # 可以调整：32/64/128
        output_channels=9,
    )
    return student
```

### Step 2: 训练蒸馏模型

```bash
python3 train_distillation.py \
    --teacher_checkpoint ../../../../../logs/best_discrete_gestures.pt \
    --data_dir /path/to/emg_data \
    --split_csv /path/to/split.csv \
    --output_dir ./ultra_light_models \
    --max_epochs 100 \
    --temperature 4.0 \
    --alpha 0.6
```

**注意**：ConvOnly模型更简单，建议：
- 提高temperature到4.0（更柔和的蒸馏）
- 提高alpha到0.6-0.7（更依赖Teacher）

### Step 3: 量化为INT8

```bash
# 运行量化脚本
python3 quantization_utils.py

# 或者自定义量化
python3 << EOF
import torch
from ultra_light_models import ConvOnlyStudentArchitecture
from quantization_utils import quantize_model_dynamic

# 加载训练好的模型
model = ConvOnlyStudentArchitecture()
model.load_state_dict(torch.load('ultra_light_models/student_final.pt'))

# 量化
quantized = quantize_model_dynamic(
    model,
    output_path='convonly_int8.pt',
    qconfig_spec={torch.nn.Conv1d, torch.nn.Linear}
)

print("✅ 量化完成，模型已缩小4倍")
EOF
```

### Step 4: 转换为TFLite

```bash
# 安装转换工具
pip install onnx onnx-tf tensorflow

# 1. 导出ONNX
python3 -c "
from quantization_utils import export_to_onnx_int8
import torch
from ultra_light_models import ConvOnlyStudentArchitecture

model = ConvOnlyStudentArchitecture()
model.load_state_dict(torch.load('convonly_int8.pt'))
export_to_onnx_int8(model, 'convonly_int8.onnx')
"

# 2. ONNX -> TensorFlow
onnx-tf convert -i convonly_int8.onnx -o convonly_tf

# 3. TensorFlow -> TFLite
python3 << EOF
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('convonly_tf')

# INT8量化配置
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 转换
tflite_model = converter.convert()

# 保存
with open('gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite模型生成完成！")
EOF
```

### Step 5: 生成C数组（用于ESP32）

```bash
# 将TFLite模型转为C头文件
xxd -i gesture_model.tflite > gesture_model.h

# 查看大小
ls -lh gesture_model.tflite
# 预期: 约80-100KB
```

### Step 6: ESP32代码

```cpp
// gesture_recognition.ino (Arduino IDE)

#include <TensorFlowLite_ESP32.h>
#include "gesture_model.h"

// TFLite相关
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 内存池（调整大小以适应模型）
constexpr int kTensorArenaSize = 100 * 1024;  // 100KB
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);
  
  // 加载模型
  model = tflite::GetModel(gesture_model_tflite);
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("模型版本不匹配!");
    return;
  }
  
  // 创建解释器
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddReLU();
  resolver.AddQuantize();
  resolver.AddDequantize();
  // ... 添加其他需要的操作
  
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );
  interpreter = &static_interpreter;
  
  // 分配内存
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("内存分配失败!");
    return;
  }
  
  // 获取输入输出张量
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("✅ 模型加载成功!");
  Serial.printf("输入shape: [%d, %d, %d]\n", 
    input->dims->data[0], 
    input->dims->data[1], 
    input->dims->data[2]
  );
}

void loop() {
  // 1. 读取EMG数据（假设从ADC读取16通道）
  float emg_data[16][2000];
  read_emg_data(emg_data);
  
  // 2. 填充输入张量（INT8量化）
  for (int c = 0; c < 16; c++) {
    for (int t = 0; t < 2000; t++) {
      // 量化：float -> int8
      int8_t quantized = (int8_t)(emg_data[c][t] * input->params.scale + input->params.zero_point);
      input->data.int8[c * 2000 + t] = quantized;
    }
  }
  
  // 3. 运行推理
  unsigned long start = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("推理失败!");
    return;
  }
  unsigned long elapsed = micros() - start;
  
  // 4. 读取输出（9个手势的logits）
  float gesture_probs[9];
  for (int i = 0; i < 9; i++) {
    // 反量化：int8 -> float
    int8_t quantized_output = output->data.int8[i];
    gesture_probs[i] = (quantized_output - output->params.zero_point) * output->params.scale;
  }
  
  // 5. 找到最可能的手势
  int max_idx = 0;
  float max_prob = gesture_probs[0];
  for (int i = 1; i < 9; i++) {
    if (gesture_probs[i] > max_prob) {
      max_prob = gesture_probs[i];
      max_idx = i;
    }
  }
  
  // 6. 输出结果
  const char* gestures[] = {
    "index_press", "index_release",
    "middle_press", "middle_release",
    "thumb_click", "thumb_down",
    "thumb_in", "thumb_out", "thumb_up"
  };
  
  Serial.printf("检测到: %s (置信度: %.2f)\n", gestures[max_idx], max_prob);
  Serial.printf("推理时间: %lu us\n", elapsed);
  
  delay(100);
}
```

## 🔧 优化技巧

### 1. 进一步减小模型

如果80KB还是太大：

```python
# ultra_light_models.py
class TinyConvOnlyArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        # 更少通道: 64 -> 32
        # 移除一层卷积
        # 预计: ~4万参数，INT8约40KB
```

### 2. 内存优化

```cpp
// 减小tensor_arena大小
constexpr int kTensorArenaSize = 80 * 1024;  // 80KB

// 使用流式推理（分批处理长时间序列）
```

### 3. 速度优化

```python
# 使用更大的stride
stride = 20  # 从10改为20，输出减半，速度翻倍
```

## 📊 性能预期

### ConvOnly + INT8 on ESP32

- **模型大小**: 80-100KB
- **RAM占用**: ~150KB（包括tensor arena）
- **推理时间**: 50-100ms（1秒@2000Hz输入）
- **准确率**: Teacher的70-80%（权衡）

### 精度损失分析

```
Teacher (650万参数):      准确率 95%
↓ 蒸馏
Student (60万参数):       准确率 90%
↓ 压缩
TinyStudent (15万参数):   准确率 85%
↓ 极致压缩
ConvOnly (8万参数):       准确率 75-80%
↓ 量化INT8
ConvOnly INT8:            准确率 75-78% (损失<2%)
```

## ⚠️ 常见问题

### Q1: ESP32内存不够？

```
方案1: 使用PSRAM（ESP32-WROVER）
方案2: 进一步减小hidden_channels到32
方案3: 使用流式推理，分段处理
```

### Q2: 推理太慢？

```
方案1: 增大stride（20或40）
方案2: 减少输入时间长度（1秒改为0.5秒）
方案3: 使用ESP32-S3（更快的CPU）
```

### Q3: 精度太低？

```
方案1: 用TinyStudent代替ConvOnly（15万参数）
方案2: 增加hidden_channels到128
方案3: 使用混合精度（部分FP16）
```

## 🎓 下一步

1. **测试ConvOnly模型**: `python3 ultra_light_models.py`
2. **开始蒸馏训练**: 修改`train_distillation.py`使用ConvOnly
3. **量化模型**: `python3 quantization_utils.py`
4. **转换为TFLite**: 按照Step 4操作
5. **ESP32部署**: 按照Step 6编写代码

祝部署顺利！🚀
