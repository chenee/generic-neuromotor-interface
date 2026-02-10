// Fake ESP SEMG filter implementation for desktop validation (no ESP-IDF dependencies)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BLOCK_SIZE 20
#define NUM_CHANNELS 16

// Filter state structure (streaming)
typedef struct {
    float hpf_delay[2];
    float lpf_delay[2];
    float notch_delays[20]; // notch_count * 2 (max 10 notches)
} FilterState;

// Processor structure
typedef struct {
    float fs;

    float hpf_coeffs[5];
    float lpf_coeffs[5];

    int notch_count;
    float **notch_coeffs;

    float lowpass;
    float highpass;
    float notch_freq_base;
    int notch_harmonics;

    float temp_block[BLOCK_SIZE];
    float hpf_output[BLOCK_SIZE];
    float lpf_output[BLOCK_SIZE];

    FilterState channel_states[NUM_CHANNELS];
} ESPSEMGProcessor;

static void filter_state_reset(FilterState *state, int notch_count) {
    if (!state) return;
    state->hpf_delay[0] = 0.0f;
    state->hpf_delay[1] = 0.0f;
    state->lpf_delay[0] = 0.0f;
    state->lpf_delay[1] = 0.0f;
    std::memset(state->notch_delays, 0, notch_count * 2 * sizeof(float));
}

// RBJ cookbook biquad generators (b0,b1,b2,a1,a2)
static void biquad_gen_hpf(float *coeffs, float fs, float fc, float q) {
    float w0 = 2.0f * (float)M_PI * fc / fs;
    float c = std::cos(w0);
    float s = std::sin(w0);
    float alpha = s / (2.0f * q);

    float b0 = (1.0f + c) / 2.0f;
    float b1 = -(1.0f + c);
    float b2 = (1.0f + c) / 2.0f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * c;
    float a2 = 1.0f - alpha;

    coeffs[0] = b0 / a0;
    coeffs[1] = b1 / a0;
    coeffs[2] = b2 / a0;
    coeffs[3] = a1 / a0;
    coeffs[4] = a2 / a0;
}

static void biquad_gen_lpf(float *coeffs, float fs, float fc, float q) {
    float w0 = 2.0f * (float)M_PI * fc / fs;
    float c = std::cos(w0);
    float s = std::sin(w0);
    float alpha = s / (2.0f * q);

    float b0 = (1.0f - c) / 2.0f;
    float b1 = 1.0f - c;
    float b2 = (1.0f - c) / 2.0f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * c;
    float a2 = 1.0f - alpha;

    coeffs[0] = b0 / a0;
    coeffs[1] = b1 / a0;
    coeffs[2] = b2 / a0;
    coeffs[3] = a1 / a0;
    coeffs[4] = a2 / a0;
}

static void biquad_gen_notch(float *coeffs, float fs, float fc, float q) {
    float w0 = 2.0f * (float)M_PI * fc / fs;
    float c = std::cos(w0);
    float s = std::sin(w0);
    float alpha = s / (2.0f * q);

    float b0 = 1.0f;
    float b1 = -2.0f * c;
    float b2 = 1.0f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * c;
    float a2 = 1.0f - alpha;

    coeffs[0] = b0 / a0;
    coeffs[1] = b1 / a0;
    coeffs[2] = b2 / a0;
    coeffs[3] = a1 / a0;
    coeffs[4] = a2 / a0;
}

static void biquad_process(const float *input, float *output, int length, const float *coeffs, float *delay) {
    const float b0 = coeffs[0];
    const float b1 = coeffs[1];
    const float b2 = coeffs[2];
    const float a1 = coeffs[3];
    const float a2 = coeffs[4];

    float d1 = delay[0];
    float d2 = delay[1];

    for (int i = 0; i < length; ++i) {
        float x = input[i];
        float y = b0 * x + d1;
        d1 = b1 * x - a1 * y + d2;
        d2 = b2 * x - a2 * y;
        output[i] = y;
    }

    delay[0] = d1;
    delay[1] = d2;
}

ESPSEMGProcessor* esp_semg_processor_init(float fs, float lowpass, float highpass,
                                          float notch_freq_base, int notch_harmonics,
                                          float notch_q) {
    ESPSEMGProcessor *proc = (ESPSEMGProcessor*)std::calloc(1, sizeof(ESPSEMGProcessor));
    if (!proc) return nullptr;

    proc->fs = fs;
    proc->lowpass = lowpass;
    proc->highpass = highpass;
    proc->notch_freq_base = notch_freq_base;
    proc->notch_harmonics = notch_harmonics;

    biquad_gen_hpf(proc->hpf_coeffs, fs, highpass, 0.707f);
    biquad_gen_lpf(proc->lpf_coeffs, fs, lowpass, 0.707f);

    proc->notch_coeffs = (float**)std::calloc(notch_harmonics, sizeof(float*));
    if (!proc->notch_coeffs) {
        std::free(proc);
        return nullptr;
    }

    proc->notch_count = 0;
    for (int k = 1; k <= notch_harmonics; ++k) {
        float f0 = notch_freq_base * k;
        if (f0 >= fs / 2.0f) break;
        proc->notch_coeffs[proc->notch_count] = (float*)std::calloc(5, sizeof(float));
        if (!proc->notch_coeffs[proc->notch_count]) {
            break;
        }
        biquad_gen_notch(proc->notch_coeffs[proc->notch_count], fs, f0, notch_q);
        proc->notch_count++;
    }

    for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
        filter_state_reset(&proc->channel_states[ch], proc->notch_count);
    }

    return proc;
}

void esp_semg_processor_free(ESPSEMGProcessor *proc) {
    if (!proc) return;
    for (int i = 0; i < proc->notch_count; ++i) {
        std::free(proc->notch_coeffs[i]);
    }
    std::free(proc->notch_coeffs);
    std::free(proc);
}

void esp_semg_process_channel(ESPSEMGProcessor *proc, const float *input,
                              float *bandpass_output, float *notched_output,
                              int length, FilterState *state) {
    biquad_process(input, proc->hpf_output, length, proc->hpf_coeffs, state->hpf_delay);
    biquad_process(proc->hpf_output, proc->lpf_output, length, proc->lpf_coeffs, state->lpf_delay);

    if (bandpass_output) {
        std::memcpy(bandpass_output, proc->lpf_output, length * sizeof(float));
    }

    std::memcpy(proc->temp_block, proc->lpf_output, length * sizeof(float));
    for (int i = 0; i < proc->notch_count; ++i) {
        float notch_output[BLOCK_SIZE];
        float *notch_delay = &state->notch_delays[i * 2];
        biquad_process(proc->temp_block, notch_output, length, proc->notch_coeffs[i], notch_delay);
        std::memcpy(proc->temp_block, notch_output, length * sizeof(float));
    }

    std::memcpy(notched_output, proc->temp_block, length * sizeof(float));
}

void esp_semg_process_multi_channel(ESPSEMGProcessor *proc, const float *input,
                                    float *bandpass_output, float *notched_output,
                                    int samples, int channels, FilterState *states) {
    for (int ch = 0; ch < channels; ++ch) {
        const float *channel_input = &input[ch * samples];
        float *channel_notched = &notched_output[ch * samples];
        float *channel_bandpass = bandpass_output ? &bandpass_output[ch * samples] : nullptr;
        esp_semg_process_channel(proc, channel_input, channel_bandpass, channel_notched, samples, &states[ch]);
    }
}

// --- Simple verification utilities ---
static float goertzel_power(const float *x, int n, float fs, float f0) {
    float w = 2.0f * (float)M_PI * f0 / fs;
    float coeff = 2.0f * std::cos(w);
    float s0 = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < n; ++i) {
        s0 = x[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

static void verify_filter() {
    const float fs = 2000.0f;
    const float lowpass = 850.0f;
    const float highpass = 20.0f;
    const float notch_base = 50.0f;
    const int notch_harm = 3;
    const float notch_q = 30.0f;

    ESPSEMGProcessor *proc = esp_semg_processor_init(fs, lowpass, highpass, notch_base, notch_harm, notch_q);
    if (!proc) {
        std::printf("init failed\n");
        return;
    }

    const int seconds = 2;
    const int total = (int)(fs * seconds);
    const int n = (total / BLOCK_SIZE) * BLOCK_SIZE;

    std::vector<float> input(n, 0.0f);
    std::vector<float> output(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        float t = (float)i / fs;
        input[i] =
            1.0f * std::sin(2.0f * (float)M_PI * 10.0f * t) +
            1.0f * std::sin(2.0f * (float)M_PI * 50.0f * t) +
            1.0f * std::sin(2.0f * (float)M_PI * 100.0f * t) +
            0.5f * std::sin(2.0f * (float)M_PI * 300.0f * t) +
            0.5f * std::sin(2.0f * (float)M_PI * 1000.0f * t);
    }

    FilterState state;
    filter_state_reset(&state, proc->notch_count);

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        esp_semg_process_channel(proc, &input[i], nullptr, &output[i], BLOCK_SIZE, &state);
    }

    const float f_list[] = {10, 50, 100, 300, 1000};
    std::printf("Freq(Hz) | InPower | OutPower | Attenuation(dB)\n");
    std::printf("---------------------------------------------\n");
    for (float f : f_list) {
        float pin = goertzel_power(input.data(), n, fs, f);
        float pout = goertzel_power(output.data(), n, fs, f);
        float att = 10.0f * std::log10((pout + 1e-12f) / (pin + 1e-12f));
        std::printf("%7.1f | %7.3e | %8.3e | %8.2f dB\n", f, pin, pout, att);
    }

    esp_semg_processor_free(proc);
}

int main() {
    verify_filter();
    return 0;
}
