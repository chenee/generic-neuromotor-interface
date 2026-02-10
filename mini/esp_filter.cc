/**
 * @file esp_filter.c
 * @brief SEMG信号滤波器 - ESP32优化版本
 *
 * 使用ESP-DSP库实现高性能SEMG信号滤波
 * 支持ESP32的硬件加速功能
 *
 * 功能：
 * 1. 带通滤波器（20-850Hz，使用级联的HPF和LPF）
 * 2. 陷波滤波器（50Hz及其谐波）
 * 3. 零相位滤波（filtfilt）
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "dsps_biquad.h"
#include "dsps_biquad_gen.h"

#include "esp_filter.h"

static const char *TAG = "filter";

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief 重置滤波器状态
 */
void filter_state_reset(FilterState *state, int notch_count) {
    if (state) {
        state->hpf_delay[0] = 0.0f;
        state->hpf_delay[1] = 0.0f;
        state->lpf_delay[0] = 0.0f;
        state->lpf_delay[1] = 0.0f;
        memset(state->notch_delays, 0, notch_count * 2 * sizeof(float));
    }
}


/**
 * @brief 初始化ESP SEMG处理器
 */
ESPSEMGProcessor* esp_semg_processor_init(float fs, float lowpass, float highpass,
                                          float notch_freq_base, int notch_harmonics,
                                          float notch_q) {
    ESPSEMGProcessor *proc = (ESPSEMGProcessor*)heap_caps_malloc(sizeof(ESPSEMGProcessor), MALLOC_CAP_SPIRAM);
    if (!proc) {
        ESP_LOGE(TAG, "内存分配失败");
        return NULL;
    }

    proc->fs = fs;
    proc->lowpass = lowpass;
    proc->highpass = highpass;
    proc->notch_freq_base = notch_freq_base;
    proc->notch_harmonics = notch_harmonics;

    // 生成高通滤波器系数
    float hpf_freq_norm = highpass / (fs / 2.0f);  // 归一化频率
    float hpf_q = 0.707f;  // Butterworth Q因子
    esp_err_t ret = dsps_biquad_gen_hpf_f32(proc->hpf_coeffs, hpf_freq_norm, hpf_q);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "生成高通滤波器系数失败");
        heap_caps_free(proc);
        return NULL;
    }

    // 生成低通滤波器系数
    float lpf_freq_norm = lowpass / (fs / 2.0f);  // 归一化频率
    float lpf_q = 0.707f;  // Butterworth Q因子
    ret = dsps_biquad_gen_lpf_f32(proc->lpf_coeffs, lpf_freq_norm, lpf_q);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "生成低通滤波器系数失败");
        heap_caps_free(proc);
        return NULL;
    }

    // 生成陷波滤波器系数
    proc->notch_count = 0;
    proc->notch_coeffs = (float**)malloc(notch_harmonics * sizeof(float*));
    if (!proc->notch_coeffs) {
        ESP_LOGE(TAG, "内存分配失败");
        heap_caps_free(proc);
        return NULL;
    }

    for (int k = 1; k <= notch_harmonics; k++) {
        float f0 = notch_freq_base * k;
        if (f0 >= fs / 2.0f) {
            break;
        }

        proc->notch_coeffs[proc->notch_count] = (float*)malloc(5 * sizeof(float));
        if (!proc->notch_coeffs[proc->notch_count]) {
            ESP_LOGE(TAG, "内存分配失败");
            // 清理已分配的内存
            for (int i = 0; i < proc->notch_count; i++) {
                free(proc->notch_coeffs[i]);
            }
            free(proc->notch_coeffs);
            heap_caps_free(proc);
            return NULL;
        }

        float notch_freq_norm = f0 / (fs / 2.0f);
        ret = dsps_biquad_gen_notch_f32(proc->notch_coeffs[proc->notch_count],
                                        notch_freq_norm, -40.0f, notch_q);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "生成陷波滤波器系数失败 (%.1f Hz)", f0);
            free(proc->notch_coeffs[proc->notch_count]);
            continue;
        }

        proc->notch_count++;
    }

    for(int ch = 0; ch < NUM_CHANNELS; ch++)
        filter_state_reset(&proc->channel_states[ch], proc->notch_count);

    ESP_LOGI(TAG, "ESP SEMG处理器初始化完成:");
    ESP_LOGI(TAG, "  采样频率: %.1f Hz", fs);
    ESP_LOGI(TAG, "  带通滤波器: %.1f-%.1f Hz", highpass, lowpass);
    ESP_LOGI(TAG, "  陷波滤波器数量: %d (50Hz及其谐波)", proc->notch_count);
    ESP_LOGI(TAG, "  使用ESP-DSP硬件加速");

    return proc;
}

/**
 * @brief 释放ESP SEMG处理器
 */
void esp_semg_processor_free(ESPSEMGProcessor *proc) {
    if (proc) {
        for (int i = 0; i < proc->notch_count; i++) {
            if (proc->notch_coeffs && proc->notch_coeffs[i]) {
                free(proc->notch_coeffs[i]);
            }
        }
        if (proc->notch_coeffs) {
            free(proc->notch_coeffs);
        }
        heap_caps_free(proc);
    }
}


/**
 * @brief 应用biquad滤波器（前向）
 */
void apply_biquad_forward(const float *coeffs, const float *input, float *output,
                         int length, float *delay) {
    dsps_biquad_f32(input, output, length, (float*)coeffs, delay);
}


/**
 * @brief 处理单通道SEMG信号（分段处理版本）
 *
 * 每次处理BLOCK_SIZE个数据点，支持流式处理
 *
 * @param proc SEMG处理器
 * @param input 输入信号
 * @param bandpass_output 带通滤波输出
 * @param notched_output 陷波滤波输出
 * @param length 信号长度
 * @param state 滤波器状态（用于保持连续性）
 */
void esp_semg_process_channel(ESPSEMGProcessor *proc, const float *input,
                              float *bandpass_output, float *notched_output,
                              int length, FilterState *state) {
    // 步骤1: 应用高通滤波器
    dsps_biquad_f32(input, proc->hpf_output, length,
                    proc->hpf_coeffs, state->hpf_delay);

    // 步骤2: 应用低通滤波器（级联）
    dsps_biquad_f32(proc->hpf_output, proc->lpf_output, length,
                    proc->lpf_coeffs, state->lpf_delay);

    // 复制带通结果
    if(bandpass_output)
        memcpy(bandpass_output, proc->lpf_output, length * sizeof(float));

    // 步骤3: 应用所有陷波滤波器
    memcpy(proc->temp_block, proc->lpf_output, length * sizeof(float));

    for (int i = 0; i < proc->notch_count; i++) {
        float notch_output[BLOCK_SIZE];
        float *notch_delay = &state->notch_delays[i * 2];

        dsps_biquad_f32(proc->temp_block, notch_output, length,
                        proc->notch_coeffs[i], notch_delay);

        memcpy(proc->temp_block, notch_output, length * sizeof(float));
    }

    // 复制最终结果
    memcpy(notched_output, proc->temp_block, length * sizeof(float));
}

/**
 * @brief 处理多通道SEMG信号（流式处理版本）
 *
 * 每个通道使用独立的滤波器状态，支持连续处理
 *
 * @param proc SEMG处理器
 * @param input 输入信号矩阵（行优先）
 * @param bandpass_output 带通滤波输出矩阵
 * @param notched_output 陷波滤波输出矩阵
 * @param samples 样本数
 * @param channels 通道数
 * @param states 每个通道的滤波器状态数组（长度为channels）
 */
 
void esp_semg_process_multi_channel(ESPSEMGProcessor *proc, const float *input,
                                    float *bandpass_output, float *notched_output,
                                    int samples, int channels, FilterState *states) {
    float *channel_input, *channel_bandpass = NULL, *channel_notched;
    for (int ch = 0; ch < channels; ch++) {
        // 提取单通道数据
        channel_input = (float*)&input[ch * samples];
        channel_notched = &notched_output[ch * samples];
        if(bandpass_output)
            channel_bandpass = &bandpass_output[ch * samples];

        // 处理单通道
        esp_semg_process_channel(proc, channel_input, channel_bandpass,
                                channel_notched, samples, &states[ch]);
//        ESP_LOGI(TAG, "处理通道 %d/%d 完成", ch + 1, channels);
    }
}

#if 0
/**
 * @brief 零相位滤波（filtfilt）
 */
void biquad_filtfilt(const float *coeffs, const float *input, float *output, int length) {
    float delay[2] = {0.0f, 0.0f};

    // 前向滤波
    dsps_biquad_f32(input, output, length, (float*)coeffs, delay);

    // 反向滤波
    delay[0] = 0.0f;
    delay[1] = 0.0f;
    apply_biquad_backward(coeffs, output, length, delay);
}

/**
 * @brief 应用biquad滤波器（反向）
 */
void apply_biquad_backward(const float *coeffs, float *data, int length, float *delay) {
    // 反转数据
    for (int i = 0; i < length / 2; i++) {
        float temp = data[i];
        data[i] = data[length - 1 - i];
        data[length - 1 - i] = temp;
    }

    // 应用滤波器
    float temp_output[length];
    dsps_biquad_f32(data, temp_output, length, (float*)coeffs, delay);

    // 再次反转
    for (int i = 0; i < length / 2; i++) {
        float temp = temp_output[i];
        temp_output[i] = temp_output[length - 1 - i];
        temp_output[length - 1 - i] = temp;
    }

    // 复制回原数组
    memcpy(data, temp_output, length * sizeof(float));
}




void esp_semg_process_multi_channel2(ESPSEMGProcessor *proc, const float *input,
                                    float *bandpass_output, float *notched_output,
                                    int samples, int channels, FilterState *states) {
    // 为每个通道分配临时缓冲区
    float *channel_input = (float*)malloc(samples * sizeof(float));
    float *channel_bandpass = (float*)malloc(samples * sizeof(float));
    float *channel_notched = (float*)malloc(samples * sizeof(float));

    if (!channel_input || !channel_bandpass || !channel_notched) {
        ESP_LOGE(TAG, "内存分配失败");
        if (channel_input) free(channel_input);
        if (channel_bandpass) free(channel_bandpass);
        if (channel_notched) free(channel_notched);
        return;
    }

    for (int ch = 0; ch < channels; ch++) {
        // 提取单通道数据
        for (int i = 0; i < samples; i++) {
            channel_input[i] = input[i * channels + ch];
        }

        // 处理单通道（使用该通道的状态）
        esp_semg_process_channel(proc, channel_input, channel_bandpass,
                                channel_notched, samples, &states[ch]);

        // 写回结果
        for (int i = 0; i < samples; i++) {
            bandpass_output[i * channels + ch] = channel_bandpass[i];
            notched_output[i * channels + ch] = channel_notched[i];
        }

        ESP_LOGI(TAG, "处理通道 %d/%d 完成", ch + 1, channels);
    }

    free(channel_input);
    free(channel_bandpass);
    free(channel_notched);
}

/**
 * @brief 处理单通道SEMG信号（批量处理版本，使用filtfilt）
 *
 * 用于离线处理，使用零相位滤波
 *
 * @param proc SEMG处理器
 * @param input 输入信号
 * @param bandpass_output 带通滤波输出
 * @param notched_output 陷波滤波输出
 * @param length 信号长度
 */
void esp_semg_process_channel_batch(ESPSEMGProcessor *proc, const float *input,
                                    float *bandpass_output, float *notched_output,
                                    int length) {
    // 临时缓冲区
    float *temp = (float*)malloc(length * sizeof(float));
    if (!temp) {
        ESP_LOGE(TAG, "内存分配失败");
        return;
    }

    // 步骤1: 应用高通滤波器
    biquad_filtfilt(proc->hpf_coeffs, input, temp, length);

    // 步骤2: 应用低通滤波器（级联）
    biquad_filtfilt(proc->lpf_coeffs, temp, bandpass_output, length);

    // 步骤3: 复制带通结果到陷波输入
    memcpy(notched_output, bandpass_output, length * sizeof(float));

    // 步骤4: 应用所有陷波滤波器
    for (int i = 0; i < proc->notch_count; i++) {
        biquad_filtfilt(proc->notch_coeffs[i], notched_output, temp, length);
        memcpy(notched_output, temp, length * sizeof(float));
    }

    free(temp);
}

/**
 * @brief 处理多通道SEMG信号（批量处理版本）
 *
 * 用于离线处理，使用零相位滤波，不需要状态
 *
 * @param proc SEMG处理器
 * @param input 输入信号矩阵（行优先）
 * @param bandpass_output 带通滤波输出矩阵
 * @param notched_output 陷波滤波输出矩阵
 * @param samples 样本数
 * @param channels 通道数
 */
void esp_semg_process_multi_channel_batch(ESPSEMGProcessor *proc, const float *input,
                                          float *bandpass_output, float *notched_output,
                                          int samples, int channels) {
    // 为每个通道分配临时缓冲区
    float *channel_input = (float*)malloc(samples * sizeof(float));
    float *channel_bandpass = (float*)malloc(samples * sizeof(float));
    float *channel_notched = (float*)malloc(samples * sizeof(float));

    if (!channel_input || !channel_bandpass || !channel_notched) {
        ESP_LOGE(TAG, "内存分配失败");
        if (channel_input) free(channel_input);
        if (channel_bandpass) free(channel_bandpass);
        if (channel_notched) free(channel_notched);
        return;
    }

    for (int ch = 0; ch < channels; ch++) {
        // 提取单通道数据
        for (int i = 0; i < samples; i++) {
            channel_input[i] = input[i * channels + ch];
        }

        // 处理单通道（批量处理，使用filtfilt）
        esp_semg_process_channel_batch(proc, channel_input, channel_bandpass,
                                      channel_notched, samples);

        // 写回结果
        for (int i = 0; i < samples; i++) {
            bandpass_output[i * channels + ch] = channel_bandpass[i];
            notched_output[i * channels + ch] = channel_notched[i];
        }

        ESP_LOGI(TAG, "处理通道 %d/%d 完成", ch + 1, channels);
    }

    free(channel_input);
    free(channel_bandpass);
    free(channel_notched);
}
#endif
