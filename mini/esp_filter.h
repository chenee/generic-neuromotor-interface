/**
 * @file esp_filter.h
 * @brief ESP32 SEMG信号滤波器头文件
 *
 * 使用ESP-DSP库实现高性能SEMG信号滤波
 */

#ifndef ESP_FILTER_H
#define ESP_FILTER_H

#include "esp_dsp.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCK_SIZE 20     // 每次处理的数据块大小
#define NUM_CHANNELS 16   // 通道数量

// 滤波器状态结构（用于流式处理）
typedef struct {
    float hpf_delay[2];           // 高通滤波器延迟状态
    float lpf_delay[2];           // 低通滤波器延迟状态
    float notch_delays[20];          // 陷波滤波器延迟状态数组 (notch_count * 2)
} FilterState;

// SEMG处理器结构
typedef struct {
    float fs;                     // 采样频率

    // 带通滤波器（使用HPF + LPF实现）
    float hpf_coeffs[5];          // 高通滤波器系数
    float lpf_coeffs[5];          // 低通滤波器系数

    // 陷波滤波器
    int notch_count;              // 陷波滤波器数量
    float **notch_coeffs;         // 陷波滤波器系数数组

    float lowpass;                // 低通截止频率
    float highpass;               // 高通截止频率
    float notch_freq_base;        // 陷波基频
    int notch_harmonics;          // 陷波谐波数

    // 临时缓冲区
    float temp_block[BLOCK_SIZE];
    float hpf_output[BLOCK_SIZE];
    float lpf_output[BLOCK_SIZE];

    // 16个通道的滤波器状态（集成在处理器中）
    FilterState channel_states[NUM_CHANNELS];
} ESPSEMGProcessor;

/**
 * @brief 初始化ESP SEMG处理器
 *
 * @param fs 采样频率 (Hz)
 * @param lowpass 低通截止频率 (Hz)
 * @param highpass 高通截止频率 (Hz)
 * @param notch_freq_base 陷波基频 (Hz)
 * @param notch_harmonics 陷波谐波数量
 * @param notch_q 陷波Q因子
 * @return ESPSEMGProcessor* 处理器指针，失败返回NULL
 */
ESPSEMGProcessor* esp_semg_processor_init(float fs, float lowpass, float highpass,
                                          float notch_freq_base, int notch_harmonics,
                                          float notch_q);

/**
 * @brief 释放ESP SEMG处理器
 *
 * @param proc 处理器指针
 */
void esp_semg_processor_free(ESPSEMGProcessor *proc);

/**
 * @brief 重置所有通道的滤波器状态
 *
 * @param proc 处理器指针
 */
void esp_semg_processor_reset_states(ESPSEMGProcessor *proc);

/**
 * @brief 处理单通道SEMG信号（流式处理）
 *
 * 每次处理BLOCK_SIZE个数据点，支持连续流式处理
 *
 * @param proc SEMG处理器
 * @param channel_index 通道索引 (0-15)
 * @param input 输入信号
 * @param bandpass_output 带通滤波输出
 * @param notched_output 陷波滤波输出
 * @param length 信号长度
 */
void esp_semg_process_channel(ESPSEMGProcessor *proc, const float *input, float *bandpass_output,
                              float *notched_output, int length, FilterState *state);

/**
 * @brief 处理多通道SEMG信号（流式处理）
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
                                   int samples, int channels, FilterState *states);

                                   /**
 * @brief 处理单通道SEMG信号（批量处理）
 *
 * 用于离线处理，使用零相位滤波（filtfilt）
 *
 * @param proc SEMG处理器
 * @param input 输入信号
 * @param bandpass_output 带通滤波输出
 * @param notched_output 陷波滤波输出
 * @param length 信号长度
 */
void esp_semg_process_channel_batch(ESPSEMGProcessor *proc, const float *input,
                                   float *bandpass_output, float *notched_output,
                                   int length);

/**
 * @brief 处理多通道SEMG信号（批量处理）
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
                                         int samples, int channels);

#ifdef __cplusplus
}
#endif

#endif // ESP_FILTER_H
