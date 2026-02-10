#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_private/esp_clk.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <inttypes.h>
#include <string.h>

// 引入 MinGRU-XS 模型数据
#include "esp_spiffs.h"
#include "esp_task_wdt.h"
#include "model_data.h"
#include <math.h>

// 辅助函数：Sigmoid
inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static const char *TAG = "MinLSTM_XS_S20K32";

// 验证数据的结构体
struct VerifHeader {
  uint32_t magic;
  uint32_t num_steps;
  uint32_t in_ch;
  uint32_t out_ch;
  uint32_t raw_len; // <--- 必须补上这个
};

// 使用 MinLSTM_XS_S20K32 模型
#define CURRENT_MODEL_DATA create_model_MinLSTM_XS_S20K32_stateful_tflite

// ⚠️ 关键修改：设置 Arena 大小
// 将 Arena 大小减小以便尽可能放入 Internal SRAM（速度最快）
// 注意：如果 AllocateTensors 失败，请适当增大该值
const int kTensorArenaSize = 60 * 1024; // 60KB
uint8_t *tensor_arena = nullptr;
uint8_t *model_in_ram = nullptr; // 用于存储从 Flash 拷贝到 RAM 的模型数据

// Helper to check memory location
bool ptr_is_in_internal_ram(void *ptr) {
  return ((uint32_t)ptr >= 0x3FC80000 && (uint32_t)ptr < 0x3FD00000) ||
         ((uint32_t)ptr >= 0x40370000 && (uint32_t)ptr < 0x403E0000);
}

extern "C" void app_main(void) {
  // 1. 初始化并打印详细设备信息
  ESP_LOGI(TAG, "==================================================");
  ESP_LOGI(TAG, "      ESP32-S3 TFLite-Micro 基准测试程序");
  ESP_LOGI(TAG, "==================================================");

  // 芯片信息
  esp_chip_info_t chip_info;
  esp_chip_info(&chip_info);

  // Increase Watchdog timeout to 60s to handle slow float32 inference
  esp_task_wdt_config_t twdt_config = {
      .timeout_ms = 60000,
      .idle_core_mask = (1 << 0) | (1 << 1), // Subscribe both idle tasks
      .trigger_panic = false,
  };
  esp_task_wdt_reconfigure(&twdt_config);

  ESP_LOGI(TAG, "设备信息:");
  ESP_LOGI(TAG, "  芯片型号:   %s", CONFIG_IDF_TARGET);
  ESP_LOGI(TAG, "  核心数量:   %d", chip_info.cores);
  ESP_LOGI(TAG, "  芯片修订:   v%d.%d", chip_info.revision / 100,
           chip_info.revision % 100);

  // CPU 频率
  uint32_t cpu_freq = esp_clk_cpu_freq();
  ESP_LOGI(TAG, "  CPU 频率:   %lu MHz", cpu_freq / 1000000);

  // 内存信息
  ESP_LOGI(TAG, "内存状态:");
  size_t free_sram =
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  size_t total_sram =
      heap_caps_get_total_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  ESP_LOGI(TAG, "  内部 SRAM:  可用 %6.1f KB / 总计 %6.1f KB",
           free_sram / 1024.0f, total_sram / 1024.0f);

  size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
  if (total_psram > 0) {
    ESP_LOGI(TAG, "  外部 PSRAM: 可用 %6.1f KB / 总计 %6.1f KB",
             free_psram / 1024.0f, total_psram / 1024.0f);
  } else {
    ESP_LOGI(TAG, "  外部 PSRAM: 未检测到或未初始化 (建议在 menuconfig "
                  "中开启以支持更大模型)");
  }
  ESP_LOGI(TAG, "--------------------------------------------------");

  // 2. 将模型拷贝到 SRAM 以提高速度
  // ESP32-S3 SRAM 充足，尽量放 SRAM。如果不行 (模型巨大)，可以考虑放 PSRAM。
  size_t model_size = sizeof(CURRENT_MODEL_DATA);

  // 4. Load Model into RAM (Try Internal SRAM first, then PSRAM)
  // 270KB Model + 90KB Arena ≈ 360KB (Close to 370KB limit)
  ESP_LOGI(TAG, "Allocating memory for Model (%d bytes)...",
           (unsigned int)model_size);
  // model_in_ram = (uint8_t *)heap_caps_malloc(model_size, MALLOC_CAP_INTERNAL |
  //                                                            MALLOC_CAP_8BIT);

  if (model_in_ram == nullptr) {
    ESP_LOGW(
        TAG,
        "⚠️ Internal SRAM insufficient for Model, falling back to PSRAM...");
    model_in_ram = (uint8_t *)heap_caps_malloc(model_size, MALLOC_CAP_SPIRAM |
                                                               MALLOC_CAP_8BIT);
  }

  if (model_in_ram == nullptr) {
    ESP_LOGE(TAG, "❌ Failed to allocate memory for Model (even in PSRAM)!");
    return;
  }

  // Copy to allocated RAM
  memcpy(model_in_ram, CURRENT_MODEL_DATA, model_size);
  ESP_LOGI(TAG, "✓ Using Model Data at %p (%s)", model_in_ram,
           (ptr_is_in_internal_ram(model_in_ram) ? "Internal SRAM" : "PSRAM"));
  if (heap_caps_check_integrity_all(true)) {
    // Check passed
  }

  // 3. 动态分配 Arena
  // 优先分配在内部 SRAM 为了速度
  // tensor_arena = (uint8_t *)heap_caps_malloc(
  //     kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

  // Fallback 到 PSRAM
  if (tensor_arena == nullptr && total_psram > 0) {
    ESP_LOGW(TAG, "⚠️ 内部 SRAM 不足以为 Arena 分配 %d KB，尝试 PSRAM...",
             kTensorArenaSize / 1024);
    tensor_arena =
        (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
  }

  if (tensor_arena == nullptr) {
    ESP_LOGE(TAG, "❌ 错误: 无法为 Tensor Arena 分配 %d 字节内存!",
             kTensorArenaSize);
    return;
  }
  ESP_LOGI(TAG, "✓ 已分配 Tensor Arena: %d 字节 (%.1f KB)", kTensorArenaSize,
           kTensorArenaSize / 1024.0f);

  // 使用 RAM 中的模型数据指针
  const tflite::Model *model = tflite::GetModel(model_in_ram);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "❌ 错误: 模型 Schema 版本不匹配!");
    return;
  }
  ESP_LOGI(TAG, "✓ 模型加载成功");

  // 2. 注册算子
  // ESP32-S3 性能较强，可以使用
  // AllOpsResolver，但为了保持二进制大小可控，通常还是推荐 MutableOpResolver
  static tflite::MicroMutableOpResolver<64> resolver; // 增加一些容量

  // 基础数学运算
  resolver.AddAdd();
  resolver.AddSub();
  resolver.AddMul();
  resolver.AddDiv();
  resolver.AddNeg();
  resolver.AddExp();
  resolver.AddMaximum();
  resolver.AddMinimum();
  resolver.AddAbs();
  resolver.AddMean();
  resolver.AddSquaredDifference();
  resolver.AddRsqrt(); // Added for new model

  // 激活函数
  resolver.AddTanh();
  resolver.AddLogistic(); // Sigmoid

  // 核心层
  resolver.AddFullyConnected();
  resolver.AddConv2D();

  // 形状处理算子
  resolver.AddReshape();
  resolver.AddUnpack();
  resolver.AddPack();
  resolver.AddTranspose();

  // SPLIT 系列
  resolver.AddSplit();
  resolver.AddSplitV();

  // 连接与切片
  resolver.AddConcatenation();
  resolver.AddStridedSlice();
  resolver.AddSlice();

  // 量化支持
  resolver.AddQuantize();
  resolver.AddDequantize();

  // 比较运算
  resolver.AddLess();
  resolver.AddGreater();

  ESP_LOGI(TAG, "Creating interpreter...");
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  tflite::MicroInterpreter *interpreter = &static_interpreter;

  ESP_LOGI(TAG, "正在分配 Tensors...");
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "❌ AllocateTensors 失败!");
    ESP_LOGE(TAG, "可能的原因:");
    ESP_LOGE(TAG, "  1. Arena 大小 (%d KB) 太小", kTensorArenaSize / 1024);
    ESP_LOGE(TAG, "  2. 模型对于当前配置来说太大");
    return;
  }

  size_t used_bytes = interpreter->arena_used_bytes();
  ESP_LOGI(TAG, "✓ Tensors 分配成功");
  ESP_LOGI(TAG, "模型诊断信息:");
  ESP_LOGI(TAG, "  模型名称:          MinLSTM_XS_S20K32 (全精度版)");
  ESP_LOGI(TAG, "  运行平台:          ESP32-S3");
  ESP_LOGI(TAG, "  模型大小:          %u 字节 (%.1f KB)",
           (unsigned int)model_size, model_size / 1024.0f);
  ESP_LOGI(TAG, "  Arena 分配容量:    %d 字节 (%.1f KB)", kTensorArenaSize,
           kTensorArenaSize / 1024.0f);
  ESP_LOGI(TAG, "  Arena 实际使用:    %d 字节 (%.1f KB)", used_bytes,
           used_bytes / 1024.0f);
  ESP_LOGI(TAG, "  Arena 剩余空间:    %d 字节 (%.1f KB)",
           kTensorArenaSize - used_bytes,
           (kTensorArenaSize - used_bytes) / 1024.0f);
  ESP_LOGI(TAG, "  当前空闲堆内存:    %lu 字节", esp_get_free_heap_size());

  TfLiteTensor *input = interpreter->input(0);
  TfLiteTensor *output = interpreter->output(0);

  ESP_LOGI(TAG, "输入输出详情:");
  ESP_LOGI(TAG, "  输入大小: %d 字节, 数据类型: %s", input->bytes,
           TfLiteTypeGetName(input->type));
  ESP_LOGI(TAG, "  输出大小: %d 字节, 数据类型: %s", output->bytes,
           TfLiteTypeGetName(output->type));
  ESP_LOGI(TAG, "--------------------------------------------------");

  // 自动适配输入数据类型
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < input->bytes / sizeof(float); ++i)
      input->data.f[i] = 0.5f;
  } else if (input->type == kTfLiteInt8) {
    for (int i = 0; i < input->bytes; ++i)
      input->data.int8[i] = 10;
  }

  ESP_LOGI(TAG, "正在进行热身 (3 次推理)...");
  for (int i = 0; i < 3; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      ESP_LOGE(TAG, "❌ 热身期间推理失败!");
      return;
    }
  }

  // --- STM32 风格基准测试逻辑 ---

  // 1. 识别 Input/Output
  // 逻辑升级: 自动匹配 State 张量，防止 Input[1] 对应 Output[0] 的情况

  TfLiteTensor *input_data = interpreter->input(0);
  TfLiteTensor *input_state = nullptr;
  TfLiteTensor *output_pred = interpreter->output(0); // 默认为 0，下面会修正
  TfLiteTensor *output_state = nullptr;

  // 尝试获取 Input State
  if (interpreter->inputs().size() > 1) {
    input_state = interpreter->input(1);
    ESP_LOGI(TAG, "Found Input State: %d bytes (Index 1)", input_state->bytes);
  }

  // 智能识别 Output
  if (interpreter->outputs().size() > 1 && input_state != nullptr) {
    TfLiteTensor *out0 = interpreter->output(0);
    TfLiteTensor *out1 = interpreter->output(1);

    // 启发式策略: 如果 Out[0] 的大小等于 State 的大小，且不等于 Out[1]
    // 的大小，则 Out[0] 可能是 State
    if (out0->bytes == input_state->bytes && out0->bytes != out1->bytes) {
      output_state = out0;
      output_pred = out1;
      ESP_LOGI(TAG, "Detected Output Layout: [0]=State, [1]=Prediction");
    } else {
      output_pred = out0;
      output_state = out1;
      ESP_LOGI(TAG, "Detected Output Layout: [0]=Prediction, [1]=State");
    }
  } else {
    // 单输出或无 State 输入的情况
    output_pred = interpreter->output(0);
    if (interpreter->outputs().size() > 1) {
      output_state = interpreter->output(1); // 盲猜
    }
    ESP_LOGI(TAG, "Default Output Layout: [0]=Prediction");
  }

  // 如果只有一个 Output 但大小和 Input State 匹配，可能 Output[0] 就是 State?
  // 但用户说 Output[0] 是 Pred (size 9 vs 684). Log output size 684. Log input
  // 13440. 我们暂时依据 Log: Only 1 Input / 1 Output shown in your log? Log:
  // Input 13440, Output 684. Wait, log says "输入大小: 13440... 输出大小: 684".
  // It loop printed input(0) and output(0). Let's print ALL inputs/outputs
  // first just to be sure.
  ESP_LOGI(TAG, "Num Inputs: %d, Num Outputs: %d", interpreter->inputs().size(),
           interpreter->outputs().size());
  for (int i = 0; i < interpreter->inputs().size(); i++) {
    ESP_LOGI(TAG, "Input[%d]: %d bytes", i, interpreter->input(i)->bytes);
  }
  for (int i = 0; i < interpreter->outputs().size(); i++) {
    ESP_LOGI(TAG, "Output[%d]: %d bytes", i, interpreter->output(i)->bytes);
  }

  // 依赖维度检测来选择 Ring Buffer 策略
  bool is_channel_major = false; // 默认 Time-Major [Batch, Time, Channels]
  int time_steps = 0;
  int channels = 0;

  // Print & Detect Layout
  ESP_LOGI(TAG, "Input Tensor Shape Checking:");
  if (input_data->dims->size >= 3) {
    // Assume Rank 3: [Batch, Dim1, Dim2]
    // usually [1, 210, 16] or [1, 16, 210]
    int dim1 = input_data->dims->data[1];
    int dim2 = input_data->dims->data[2];
    ESP_LOGI(TAG, "  Dims: [%d, %d, %d, ...]", input_data->dims->data[0], dim1,
             dim2);

    if (dim2 == 16) {
      ESP_LOGI(TAG, "  -> Detected Time-Major Layout [Batch, Time, Channels]");
      is_channel_major = false;
      time_steps = dim1;
      channels = dim2;
    } else if (dim1 == 16) {
      ESP_LOGI(TAG,
               "  -> Detected Channel-Major Layout [Batch, Channels, Time]");
      is_channel_major = true;
      time_steps = dim2;
      channels = dim1;
    } else {
      ESP_LOGW(TAG, "  -> Unknown Layout! Defaulting to Time-Major (Legacy)");
      is_channel_major = false;
    }
  } else {
    ESP_LOGW(TAG, "  -> Unexpected Rank < 3! Dims Size: %d",
             input_data->dims->size);
  }

  // STM32 Benchmark Config (Semantically Identical)
  // STM32 (Int8): 320 bytes = 320 elements
  // ESP32 (Float32): 320 elements = 20 steps * 16 channels. 320 * 4 bytes = 1280
  // bytes Real-world stride is 20 steps @ 1000Hz (20ms). Kernel width is 32
  // steps (receptive field).
  const int NUM_ITERATIONS = 100;
  const int STEPS_NEW = 20;
  const int ELEMENTS_NEW = 320;
  const int BYTES_NEW = ELEMENTS_NEW * sizeof(float); // 1280 bytes

  // For Channel Major: Shift per channel
  // Shift 10 steps. Total steps usually 210.
  int steps_total =
      (time_steps > 0) ? time_steps : 210; // Default 210 if detect fail

  int bytes_shift = input_data->bytes - BYTES_NEW;
  if (bytes_shift < 0)
    bytes_shift = 0;

  // 确保基准测试从干净的状态开始
  if (input_state) {
    if (input_state->type == kTfLiteFloat32) {
      memset(input_state->data.f, 0, input_state->bytes);
    } else {
      memset(input_state->data.uint8, 0, input_state->bytes);
    }
  }

  ESP_LOGI(TAG,
           "Starting %d Iterations (RingBuffer New: %d elements / %d bytes)...",
           NUM_ITERATIONS, ELEMENTS_NEW, BYTES_NEW);

  // 分配 Shadow Buffer 以防止 TFLM 内存复用损坏数据
  float *bench_shadow =
      (float *)heap_caps_malloc(input_data->bytes, MALLOC_CAP_SPIRAM);
  if (bench_shadow) {
    memset(bench_shadow, 0, input_data->bytes);
  } else {
    ESP_LOGE(TAG, "Failed to allocate bench shadow buffer!");
    return;
  }

  int64_t total_time = 0;

  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    int64_t start = esp_timer_get_time();

    // 1. Ring Buffer Simulation (在 Shadow Buffer 上操作)
    if (i > 0) {
      if (!is_channel_major) {
        // Time-Major
        if (bytes_shift > 0) {
          memmove(bench_shadow, (uint8_t *)bench_shadow + BYTES_NEW,
                  bytes_shift);
        }
      } else {
        // Channel-Major
        int steps_shift = STEPS_NEW; // 20
        int steps_keep = steps_total - steps_shift;
        for (int ch = 0; ch < channels; ch++) {
          float *ch_ptr = bench_shadow + (ch * steps_total);
          memmove(ch_ptr, ch_ptr + steps_shift, steps_keep * sizeof(float));
        }
      }
    }
    // [同步到 TFLite]
    memcpy(input_data->data.f, bench_shadow, input_data->bytes);

    // 2. Invoke
    if (interpreter->Invoke() != kTfLiteOk) {
      ESP_LOGE(TAG, "Invoke failed at iter %d", i);
      return;
    }

    // 3. State Copy (Output State -> Input State)
    if (input_state && output_state &&
        (input_state->data.data != output_state->data.data)) {
      memcpy(input_state->data.data, output_state->data.data,
             input_state->bytes);
    }

    int64_t end = esp_timer_get_time();
    total_time += (end - start);

    if ((i + 1) % 10 == 0) {
      ESP_LOGI(TAG, "  Progress: %d/%d", i + 1, NUM_ITERATIONS);
    }
    // Yield every iteration to prevent WDT on slow loops
    vTaskDelay(1);
  }

  ESP_LOGI(TAG, "");
  ESP_LOGI(TAG, "================ 基准测试结果 ================");
  ESP_LOGI(TAG, "设备型号:           ESP32-S3");
  ESP_LOGI(TAG, "测试模型:           MinLSTM_XS_S20K32");
  ESP_LOGI(TAG, "Total Time:         %" PRIi64 " ms", total_time / 1000);
  ESP_LOGI(TAG, "平均推理延迟:       %" PRIi64 " us (%.3f ms)",
           total_time / NUM_ITERATIONS,
           (double)total_time / NUM_ITERATIONS / 1000.0);
  ESP_LOGI(TAG, "吞吐量 (FPS):       %.2f",
           1000000.0 / ((double)total_time / NUM_ITERATIONS));
  ESP_LOGI(TAG, "==============================================");
  ESP_LOGI(TAG, "最终空闲堆内存:     %lu 字节", esp_get_free_heap_size());
  ESP_LOGI(TAG, "最小空闲堆内存:     %lu 字节",
           esp_get_minimum_free_heap_size());

  ESP_LOGI(TAG, "基准测试成功完成!");
  heap_caps_free(bench_shadow);

  // ... Benchmark Done ...

  // --- Verification Mode (SPIFFS) ---
  ESP_LOGI(TAG, "Checking for Verification Data in SPIFFS...");

  esp_vfs_spiffs_conf_t conf = {.base_path = "/storage",
                                .partition_label = "storage",
                                .max_files = 5,
                                .format_if_mount_failed = false};

  if (esp_vfs_spiffs_register(&conf) == ESP_OK) {
    ESP_LOGI(TAG, "SPIFFS mounted successfully");

    const int kModelStride = 20;
    FILE *f = fopen("/storage/verif.bin", "rb");

    if (f != NULL) {
      ESP_LOGI(TAG, "Found verif.bin, loading to PSRAM...");

      // 1. 读取头信息 (包含 raw_len)
      VerifHeader header;
      fread(&header, sizeof(VerifHeader), 1, f);

      if (header.magic != 0x46524556) {
        ESP_LOGE(TAG,
                 "Invalid Magic Number! Expected 0x46524556, got 0x%08" PRIx32,
                 header.magic);
        fclose(f);
        return;
      }

      ESP_LOGI(TAG,
               "Header: Steps=%" PRIu32 ", InCh=%" PRIu32 ", OutCh=%" PRIu32
               ", RawLen=%" PRIu32,
               header.num_steps, header.in_ch, header.out_ch, header.raw_len);

      uint32_t num_pred_steps = header.num_steps; // 预测步数 (e.g., 3978)
      uint32_t in_ch = header.in_ch;
      uint32_t out_ch = header.out_ch;
      uint32_t raw_input_len = header.raw_len; // 原始输入长度 (e.g., 39990)

      // 2. 分配内存 (注意：输入数据大小基于 raw_len，而不是 num_pred_steps)
      size_t input_total_floats = raw_input_len * in_ch;
      size_t output_total_floats = num_pred_steps * out_ch;
      size_t shadow_size = steps_total * in_ch * sizeof(float);

      // 分配在 PSRAM
      float *file_inputs = (float *)heap_caps_malloc(
          input_total_floats * sizeof(float), MALLOC_CAP_SPIRAM);
      float *file_outputs = (float *)heap_caps_malloc(
          output_total_floats * sizeof(float), MALLOC_CAP_SPIRAM);
      float *file_gt_outputs = (float *)heap_caps_malloc(
          output_total_floats * sizeof(float), MALLOC_CAP_SPIRAM);
      float *shadow_buffer =
          (float *)heap_caps_malloc(shadow_size, MALLOC_CAP_SPIRAM);

      if (!file_inputs || !file_outputs || !file_gt_outputs || !shadow_buffer) {
        ESP_LOGE(TAG, "OOM: Failed to allocate file buffers in PSRAM");
        if (file_inputs)
          heap_caps_free(file_inputs);
        if (file_outputs)
          heap_caps_free(file_outputs);
        if (file_gt_outputs)
          heap_caps_free(file_gt_outputs);
        if (shadow_buffer)
          heap_caps_free(shadow_buffer);
        fclose(f);
        return;
      }

      // 3. 读取数据 (顺序必须严格对应 Python 写入顺序)
      ESP_LOGI(TAG, "Reading Input Data (%d floats)...",
               (int)input_total_floats);
      fread(file_inputs, sizeof(float), input_total_floats, f);

      ESP_LOGI(TAG, "Reading PC Reference Output Data (%d floats)...",
               (int)output_total_floats);
      fread(file_outputs, sizeof(float), output_total_floats, f);

      ESP_LOGI(TAG, "Reading Ground Truth Data (%d floats)...",
               (int)output_total_floats);
      fread(file_gt_outputs, sizeof(float), output_total_floats, f);

      fclose(f);

      // 4. 准备统计变量
      long long *tp_gt = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);
      long long *fp_gt = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);
      long long *fn_gt = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);
      long long *tn_gt = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);

      long long *tp_pc = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);
      long long *fp_pc = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);
      long long *fn_pc = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);
      long long *tn_pc = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);

      long long *tp_pc_gt = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);
      long long *fp_pc_gt = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);
      long long *fn_pc_gt = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);
      long long *tn_pc_gt = (long long *)heap_caps_calloc(
          out_ch, sizeof(long long), MALLOC_CAP_8BIT);

      if (!tp_gt || !fp_gt || !fn_gt || !tn_gt || !tp_pc || !fp_pc || !fn_pc ||
          !tn_pc || !tp_pc_gt || !fp_pc_gt || !fn_pc_gt || !tn_pc_gt) {
        ESP_LOGE(TAG, "OOM: Failed to allocate metric counters");
        return;
      }

      double total_mae = 0.0;
      int inference_count = 0;

      // 清零状态
      if (input_state)
        memset(input_state->data.data, 0, input_state->bytes);

      // --- 步骤 0: 初始化 Shadow Buffer (第一个完整window) ---
      ESP_LOGI(TAG, "Initializing shadow buffer...");
      for (int ch = 0; ch < (int)in_ch; ch++) {
        float *ch_ptr_shadow = shadow_buffer + (ch * steps_total);
        float *ch_ptr_file = file_inputs + (ch * raw_input_len);
        memcpy(ch_ptr_shadow, ch_ptr_file, steps_total * sizeof(float));
      }

      // --- 步骤 1: 首次同步到 TFLite ---
      memcpy(input_data->data.f, shadow_buffer, shadow_size);

      // Layout check (Already available after AllocateTensors)
      bool out_channel_major = false;
      int out_dim1 = 1, out_dim2 = 1;
      if (output_pred->dims->size >= 3) {
        out_dim1 = output_pred->dims->data[1];
        out_dim2 = output_pred->dims->data[2];
        if (out_dim1 == (int)out_ch)
          out_channel_major = true;
      }

      int64_t t_start = esp_timer_get_time();
      double sum_mcu = 0, sum_pc = 0; // Removed static to ensure fresh start

      // Lambda for recording metrics to avoid code duplication and off-by-one
      // errors
      auto record_metrics = [&](int v_idx) {
        float *model_out_all = output_pred->data.f;
        for (int c = 0; c < (int)out_ch; c++) {
          // MCU Output (Logit -> Sigmoid)
          float logit;
          if (out_channel_major) {
            logit = model_out_all[c * out_dim2 + (out_dim2 - 1)];
          } else {
            logit = model_out_all[(out_dim1 - 1) * out_ch + c];
          }
          float mcu_prob = sigmoid(logit);
          int mcu_bin = (mcu_prob > 0.35f) ? 1 : 0;

          // PC Reference (Already Prob)
          float pc_prob = file_outputs[c * num_pred_steps + v_idx];
          int pc_bin = (pc_prob > 0.35f) ? 1 : 0;

          // GT (Binary)
          float gt_val = file_gt_outputs[c * num_pred_steps + v_idx];
          int gt_bin = (gt_val > 0.35f) ? 1 : 0;

          // Stats
          total_mae += fabsf(mcu_prob - pc_prob);
          sum_mcu += mcu_prob;
          sum_pc += pc_prob;

          // Confusion Matrix: MCU vs GT
          if (gt_bin == 1) {
            if (mcu_bin == 1)
              tp_gt[c]++;
            else
              fn_gt[c]++;
          } else {
            if (mcu_bin == 1)
              fp_gt[c]++;
            else
              tn_gt[c]++;
          }
          // Confusion Matrix: MCU vs PC
          if (pc_bin == 1) {
            if (mcu_bin == 1)
              tp_pc[c]++;
            else
              fn_pc[c]++;
          } else {
            if (mcu_bin == 1)
              fp_pc[c]++;
            else
              tn_pc[c]++;
          }
          // Confusion Matrix: PC vs GT
          if (gt_bin == 1) {
            if (pc_bin == 1)
              tp_pc_gt[c]++;
            else
              fn_pc_gt[c]++;
          } else {
            if (pc_bin == 1)
              fp_pc_gt[c]++;
            else
              tn_pc_gt[c]++;
          }
        }
      };

      // Warmup Invoke
      if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Initial warmup invoke failed!");
        return;
      }
      // Warmup State Update
      if (input_state && output_state) {
        memcpy(input_state->data.data, output_state->data.data,
               input_state->bytes);
      }

      // === [关键修复] 记录 Warmup 输出作为第 0 条 ===
      record_metrics(0);
      int verif_idx = 1;
      inference_count = 1;

      // 循环逻辑：i 代表原始数据的偏移量，执行剩余的 num_steps - 1 次
      int remaining = num_pred_steps - 1;
      for (int loop = 0; loop < remaining; ++loop) {
        int i = kModelStride + loop * kModelStride;

        // 确保没有越界访问验证数组
        if (verif_idx >= (int)num_pred_steps)
          break;

        // --- 步骤 A: Ring Buffer Update (在 Shadow Buffer 上) ---
        for (int ch = 0; ch < (int)in_ch; ch++) {
          float *shadow_ch_ptr = shadow_buffer + (ch * steps_total);
          // Shift
          memmove(shadow_ch_ptr, shadow_ch_ptr + kModelStride,
                  (steps_total - kModelStride) * sizeof(float));
          // Append
          float *src_new_data = file_inputs + (ch * raw_input_len) +
                                (steps_total + i - kModelStride);
          memcpy(shadow_ch_ptr + (steps_total - kModelStride), src_new_data,
                 kModelStride * sizeof(float));
        }

        // --- 步骤 B: 将干净的数据拷贝给 TFLite ---
        memcpy(input_data->data.f, shadow_buffer, shadow_size);

        // --- 步骤 C: Invoke ---
        if (interpreter->Invoke() != kTfLiteOk)
          break;

        // --- 步骤 D: Update State ---
        if (input_state && output_state) {
          memcpy(input_state->data.data, output_state->data.data,
                 input_state->bytes);
        }

        // --- 步骤 E: Metrics ---
        record_metrics(verif_idx);

        if (inference_count % 500 == 0) {
          ESP_LOGI(TAG, "Verification progress: %d/%u (MAE: %.6f)",
                   inference_count, (unsigned int)num_pred_steps,
                   total_mae / (inference_count * out_ch));
        }

        verif_idx++;
        inference_count++;

        // Yield every iteration to prevent WDT on slow loops
        vTaskDelay(1);
      }

      int64_t t_end = esp_timer_get_time();

      // --- 报告生成 ---
      ESP_LOGI(TAG, "========================================");
      ESP_LOGI(TAG, "Multi-label Verification Results (Threshold 0.35)");
      ESP_LOGI(TAG, "Total Inferences: %d", inference_count);
      ESP_LOGI(TAG, "Probability MAE (vs PyTorch): %.6f",
               total_mae / (inference_count * out_ch));
      ESP_LOGI(TAG, "Avg MCU Prob: %.6f", sum_mcu / (inference_count * out_ch));
      ESP_LOGI(TAG, "Avg PC Prob: %.6f", sum_pc / (inference_count * out_ch));
      ESP_LOGI(TAG, "Prob Bias (MCU - PC): %.6f",
               (sum_mcu - sum_pc) / (inference_count * out_ch));
      ESP_LOGI(TAG, "Time Taken:      %.2f s", (t_end - t_start) / 1000000.0);
      ESP_LOGI(TAG, "----------------------------------------");

      auto print_metrics = [&](const char *title, const char *metric_name,
                               long long *tp, long long *fp, long long *fn,
                               long long *tn) {
        long long total_tp = 0, total_fp = 0, total_fn = 0, total_tn = 0;
        for (int c = 0; c < (int)out_ch; c++) {
          total_tp += tp[c];
          total_fp += fp[c];
          total_fn += fn[c];
          total_tn += tn[c];
        }

        double overall_acc =
            (double)(total_tp + total_tn) / (inference_count * out_ch);
        double overall_prec = (total_tp + total_fp > 0)
                                  ? (double)total_tp / (total_tp + total_fp)
                                  : 0.0;
        double overall_rec = (total_tp + total_fn > 0)
                                 ? (double)total_tp / (total_tp + total_fn)
                                 : 0.0;
        double overall_f1 =
            (overall_prec + overall_rec > 0)
                ? 2 * overall_prec * overall_rec / (overall_prec + overall_rec)
                : 0.0;

        ESP_LOGI(TAG, "Overall (Flattened) %s:", title);
        ESP_LOGI(TAG, "  %s: %.4f (%.2f%%)", metric_name, (1.0 - overall_rec),
                 (1.0 - overall_rec) * 100);
        ESP_LOGI(TAG, "  Accuracy:  %.4f", overall_acc);
        ESP_LOGI(TAG, "  Precision: %.4f", overall_prec);
        ESP_LOGI(TAG, "  Recall:    %.4f", overall_rec);
        ESP_LOGI(TAG, "  F1-Score:  %.4f", overall_f1);
        ESP_LOGI(TAG, "  TP=%lld, FP=%lld, FN=%lld, TN=%lld", total_tp,
                 total_fp, total_fn, total_tn);

        ESP_LOGI(TAG, "Per-Gesture Metrics:");
        for (int c = 0; c < (int)out_ch; c++) {
          double acc = (double)(tp[c] + tn[c]) / inference_count;
          double prec =
              (tp[c] + fp[c] > 0) ? (double)tp[c] / (tp[c] + fp[c]) : 0.0;
          double rec =
              (tp[c] + fn[c] > 0) ? (double)tp[c] / (tp[c] + fn[c]) : 0.0;
          double f1 = (prec + rec > 0) ? 2 * prec * rec / (prec + rec) : 0.0;
          ESP_LOGI(TAG,
                   "  G%d: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f [TP=%lld, "
                   "FP=%lld, FN=%lld]",
                   c, acc, prec, rec, f1, tp[c], fp[c], fn[c]);
        }
        ESP_LOGI(TAG, "----------------------------------------");
      };

      print_metrics("MCU vs GT", "CLER", tp_gt, fp_gt, fn_gt, tn_gt);
      print_metrics("MCU vs PC Agreement", "Mismatch Rate", tp_pc, fp_pc, fn_pc,
                    tn_pc);
      print_metrics("PC vs GT", "Ref CLER", tp_pc_gt, fp_pc_gt, fn_pc_gt,
                    tn_pc_gt);

      auto print_markdown_matrix = [&](const char *title, long long *tp,
                                       long long *fp, long long *fn,
                                       long long *tn) {
        long long stp = 0, sfp = 0, sfn = 0, stn = 0;
        for (int c = 0; c < (int)out_ch; c++) {
          stp += tp[c];
          sfp += fp[c];
          sfn += fn[c];
          stn += tn[c];
        }
        ESP_LOGI(TAG, "\n### Confusion Matrix: %s", title);
        ESP_LOGI(TAG, "| | Predict Pos | Predict Neg |");
        ESP_LOGI(TAG, "| :--- | :---: | :---: |");
        ESP_LOGI(TAG, "| **Actual Pos** | %lld (TP) | %lld (FN) |", stp, sfn);
        ESP_LOGI(TAG, "| **Actual Neg** | %lld (FP) | %lld (TN) |", sfp, stn);
        ESP_LOGI(TAG, "\n");
      };

      print_markdown_matrix("MCU vs GT", tp_gt, fp_gt, fn_gt, tn_gt);
      print_markdown_matrix("MCU vs PyTorch", tp_pc, fp_pc, fn_pc, tn_pc);
      print_markdown_matrix("PyTorch vs GT", tp_pc_gt, fp_pc_gt, fn_pc_gt,
                            tn_pc_gt);

      ESP_LOGI(TAG, "========================================");

      // --- Final Memory Cleanup ---
      heap_caps_free(file_inputs);
      heap_caps_free(file_outputs);
      heap_caps_free(file_gt_outputs);
      heap_caps_free(shadow_buffer);
      heap_caps_free(tp_gt);
      heap_caps_free(fp_gt);
      heap_caps_free(fn_gt);
      heap_caps_free(tn_gt);
      heap_caps_free(tp_pc);
      heap_caps_free(fp_pc);
      heap_caps_free(fn_pc);
      heap_caps_free(tn_pc);
      heap_caps_free(tp_pc_gt);
      heap_caps_free(fp_pc_gt);
      heap_caps_free(fn_pc_gt);
      heap_caps_free(tn_pc_gt);

      while (1)
        vTaskDelay(1000);
    }
  }

  // --- Optimized True Streaming Inference ---
  ESP_LOGI(TAG, "Entering Optimized Streaming Mode...");

  // ===================== 1. 配置参数 =====================
  // 必须与 Python 导出时的设置一致
  const int kModelInputSteps = 32;   // Kernel Size
  const int kUpdateStride = 20;      // Stride (每次更新的新数据量)
  const int kChannels = 16;
  
  // 计算需要保留的历史数据长度 (Overlap): 32 - 20 = 12
  // 这是为了做卷积所必须保留的上下文
  const int kHistorySteps = kModelInputSteps - kUpdateStride; 

    // ===================== 2. 内存分配 & 零拷贝优化 =====================
    // A. 接收新数据的 Buffer (10 * 16 floats) - 使用 PSRAM/heap_caps_malloc 确保对齐
    const int kBytesNewData = kUpdateStride * kChannels * sizeof(float);
    float *uart_recv_f = (float *)heap_caps_malloc(kBytesNewData, MALLOC_CAP_8BIT);
    if (!uart_recv_f) { ESP_LOGE(TAG, "OOM: uart_recv_f"); return; }

    // 获取并校验 TFLite 输入 Tensor 布局
    TfLiteTensor *model_input = interpreter->input(0);
    if (model_input->dims->size < 3) {
      ESP_LOGE(TAG, "Input tensor rank unexpected: %d", model_input->dims->size);
      return;
    }
    int dim_ch = model_input->dims->data[1];
    int dim_time = model_input->dims->data[2];
    if (dim_ch != kChannels || dim_time != kModelInputSteps) {
      ESP_LOGE(TAG, "❌ Model dims mismatch: expected [1,%d,%d], got [1,%d,%d]",
           kChannels, kModelInputSteps, dim_ch, dim_time);
      return;
    }

    // 零拷贝输入指针 (直接在 TFLite 输入内存上移位和写入)
    float* tflite_input_ptr = model_input->data.f;
    // 初始化为零以模拟 MCU 冷启动（与 Python 验证脚本保持一致）
    memset(tflite_input_ptr, 0, model_input->bytes);

    // 关闭 IO 缓冲以降低延迟
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);

    // 简单 DC-blocker 状态（可选，根据训练数据决定是否启用）
    float dc_x[16] = {0};
    float dc_y[16] = {0};

    // 清零 RNN 状态
    if (input_state) {
      memset(input_state->data.data, 0, input_state->bytes);
    }

    while (1) {
    // ---------------------------------------------------------
    // A. 读取新数据 (Blocking)。使用 taskYIELD() 替代 vTaskDelay 以降低相位抖动
    // ---------------------------------------------------------
    size_t total_read = 0;
    uint8_t *p_recv_u8 = (uint8_t *)uart_recv_f;
    while (total_read < (size_t)kBytesNewData) {
      size_t n = fread(p_recv_u8 + total_read, 1, kBytesNewData - total_read, stdin);
      if (n > 0) {
        total_read += n;
      } else {
        taskYIELD();
      }
    }

    float *p_new_f = uart_recv_f;

    // ---------------------------------------------------------
    // B. 零拷贝更新 Window (直接在 tflite_input_ptr 上操作)
    // 内存布局: Channel-Major [ch0: t0..t20, ch1: t0..t20 ...]
    // ---------------------------------------------------------
    for (int ch = 0; ch < kChannels; ch++) {
      float* ch_ptr = tflite_input_ptr + (ch * kModelInputSteps);

      // Shift left 保留 overlap
      memmove(ch_ptr, ch_ptr + kUpdateStride, kHistorySteps * sizeof(float));

      // Append new data (De-interleave)
      for (int t = 0; t < kUpdateStride; t++) {
        float raw_val = p_new_f[t * kChannels + ch];

        // DC-blocker (可选，若训练数据未做 HPF 请移除)
        float filtered_val = raw_val - dc_x[ch] + 0.95f * dc_y[ch];
        dc_x[ch] = raw_val;
        dc_y[ch] = filtered_val;

        ch_ptr[kHistorySteps + t] = filtered_val;
      }
    }

    // ---------------------------------------------------------
    // C. 推理 (无需 memcpy)
    // ---------------------------------------------------------
    if (interpreter->Invoke() != kTfLiteOk) {
      ESP_LOGE(TAG, "Invoke failed!");
      continue;
    }

    // ---------------------------------------------------------
    // D. 状态闭环
    // ---------------------------------------------------------
    if (input_state && output_state) {
      memcpy(input_state->data.data, output_state->data.data, input_state->bytes);
    }

    // ---------------------------------------------------------
    // E. 输出结果
    // ---------------------------------------------------------
    float *out_ptr = output_pred->data.f;
    int out_count = output_pred->bytes / sizeof(float);
    for (int k = 0; k < out_count; k++) {
      out_ptr[k] = sigmoid(out_ptr[k]);
    }
    fwrite(output_pred->data.data, 1, output_pred->bytes, stdout);
    fflush(stdout);
    }
  }