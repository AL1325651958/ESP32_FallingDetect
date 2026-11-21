#include "rps_classifier.h"
#include "rps_model_data.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

static float input[128];
static float conv1[RPS_CONV1_CHANNELS * 64];
static float pooled[RPS_CONV1_CHANNELS * 16];
static float conv2[RPS_CONV2_CHANNELS * 16];
static float hidden[RPS_HIDDEN_UNITS];
#ifndef RPS_REFERENCE_KERNELS
// Shared padding workspace, reused by both convolutions (single inference task).
static float padded[(RPS_CONV1_CHANNELS * 36 > 200) ? RPS_CONV1_CHANNELS * 36 : 200];
#endif

static inline float relu_value(float value) { return value > 0.0f ? value : 0.0f; }
static inline float max_value(float a, float b) { return a > b ? a : b; }

static void conv3x3(const float *in, int ic, int oc, int size,
                    const float *weights, const float *bias, float *out, bool relu)
{
#ifdef RPS_REFERENCE_KERNELS
    for (int o = 0; o < oc; ++o) {
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                float sum = bias[o];
                for (int c = 0; c < ic; ++c) {
                    for (int ky = 0; ky < 3; ++ky) {
                        int iy = y + ky - 1;
                        if (iy < 0 || iy >= size) continue;
                        for (int kx = 0; kx < 3; ++kx) {
                            int ix = x + kx - 1;
                            if (ix < 0 || ix >= size) continue;
                            sum += in[(c * size + iy) * size + ix] *
                                   weights[((o * ic + c) * 3 + ky) * 3 + kx];
                        }
                    }
                }
                out[(o * size + y) * size + x] = relu ? fmaxf(0.0f, sum) : sum;
            }
        }
    }
#else
    const int stride = size + 2;
    const int plane = stride * stride;
    memset(padded, 0, (size_t)ic * plane * sizeof(float));
    for (int c = 0; c < ic; ++c)
        for (int y = 0; y < size; ++y)
            memcpy(padded + c * plane + (y + 1) * stride + 1,
                   in + (c * size + y) * size, (size_t)size * sizeof(float));
    for (int o = 0; o < oc; ++o) {
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                float sum = bias[o];
                const float *w = weights + o * ic * 9;
                for (int c = 0; c < ic; ++c, w += 9) {
                    const float *p = padded + c * plane + y * stride + x;
                    // No boundary checks or kernel index arithmetic in this dot product.
                    sum += p[0] * w[0];
                    sum += p[1] * w[1];
                    sum += p[2] * w[2];
                    sum += p[stride] * w[3];
                    sum += p[stride + 1] * w[4];
                    sum += p[stride + 2] * w[5];
                    sum += p[2 * stride] * w[6];
                    sum += p[2 * stride + 1] * w[7];
                    sum += p[2 * stride + 2] * w[8];
                }
                out[(o * size + y) * size + x] = relu ? relu_value(sum) : sum;
            }
        }
    }
#endif
}

#ifdef RPS_REFERENCE_KERNELS
static void residual_projection_relu(const float *in, float *out)
{
    // Unfused reference for numerical and performance comparison.
    for (int o = 0; o < RPS_CONV2_CHANNELS; ++o) {
        for (int p = 0; p < 16; ++p) {
            float shortcut = rps_skip_b[o];
            for (int c = 0; c < RPS_CONV1_CHANNELS; ++c) {
                shortcut += rps_skip_w[o * RPS_CONV1_CHANNELS + c] * in[c * 16 + p];
            }
            out[o * 16 + p] = fmaxf(0.0f, out[o * 16 + p] + shortcut);
        }
    }
}
#endif

static void dense(const float *in, int inputs, int outputs,
                  const float *weights, const float *bias, float *out, bool relu)
{
    for (int o = 0; o < outputs; ++o) {
        float sum = bias[o];
        for (int i = 0; i < inputs; ++i) sum += weights[o * inputs + i] * in[i];
        out[o] = relu ? relu_value(sum) : sum;
    }
}

void rps_predict(const int16_t distances_mm[64], rps_result_t *result)
{
    *result = (rps_result_t){.raw_label = RPS_UNKNOWN, .label = RPS_UNKNOWN};
#if RPS_INPUT_MODE != 0
    float foreground_sum = 0.0f;
    int foreground_count = 0;
    for (int i = 0; i < 64; ++i) {
        int d = distances_mm[i];
        if (d >= 0 && d < 1000) { foreground_sum += (float)d; ++foreground_count; }
    }
    float foreground_mean = foreground_count ? foreground_sum / foreground_count : 0.0f;
#endif
    for (int i = 0; i < 64; ++i) {
        int d = distances_mm[i];
        bool valid = d >= 0;
#if RPS_INPUT_MODE == 0
        input[i] = valid ? 1.0f - fminf((float)d, RPS_CLIP_MM) / RPS_CLIP_MM : 0.0f;
        input[i + 64] = valid ? 1.0f : 0.0f;
#else
        bool foreground = valid && d < 1000;
        float relative = 0.5f + (foreground_mean - (float)d) / 600.0f;
        input[i] = foreground ? fminf(1.0f, fmaxf(0.0f, relative)) : 0.0f;
        input[i + 64] = (RPS_INPUT_MODE == 2 ? foreground : valid) ? 1.0f : 0.0f;
#endif
        result->valid_zones += valid;
        result->near_zones += valid && d >= 100 && d < 1000;
    }
    conv3x3(input, 2, RPS_CONV1_CHANNELS, 8, rps_w1, rps_b1, conv1, true);
    for (int c = 0; c < RPS_CONV1_CHANNELS; ++c) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                int p = c * 64 + y * 16 + x * 2;
                pooled[c * 16 + y * 4 + x] = max_value(max_value(conv1[p], conv1[p + 1]),
                                                                     max_value(conv1[p + 8], conv1[p + 9]));
            }
        }
    }
#ifdef RPS_REFERENCE_KERNELS
    conv3x3(pooled, RPS_CONV1_CHANNELS, RPS_CONV2_CHANNELS, 4, rps_w2, rps_b2, conv2, false);
    residual_projection_relu(pooled, conv2);
#else
    conv3x3(pooled, RPS_CONV1_CHANNELS, RPS_CONV2_CHANNELS, 4, rps_w2, rps_b2, conv2, true);
#endif
    dense(conv2, RPS_CONV2_CHANNELS * 16, RPS_HIDDEN_UNITS, rps_w3, rps_b3, hidden, true);
    dense(hidden, RPS_HIDDEN_UNITS, 3, rps_w4, rps_b4, result->probabilities, false);
    float max_logit = fmaxf(result->probabilities[0], fmaxf(result->probabilities[1], result->probabilities[2]));
    float total = 0.0f;
    for (int k = 0; k < 3; ++k) { result->probabilities[k] = expf(result->probabilities[k] - max_logit); total += result->probabilities[k]; }
    for (int k = 0; k < 3; ++k) {
        result->probabilities[k] /= total;
        if (result->probabilities[k] > result->confidence) {
            result->confidence = result->probabilities[k]; result->raw_label = (rps_label_t)k;
        }
    }
    if (result->valid_zones >= 32 && result->near_zones >= 8 && result->confidence >= RPS_CONFIDENCE_THRESHOLD) {
        result->label = result->raw_label;
    }
}

const char *rps_label_name(rps_label_t label)
{
    switch (label) {
        case RPS_ROCK: return "rock";
        case RPS_PAPER: return "paper";
        case RPS_SCISSORS: return "scissors";
        default: return "unknown";
    }
}

bool rps_self_test(float *max_probability_error)
{
    float error = 0.0f;
    for (int k = 0; k < 3; ++k) {
        rps_result_t result;
        rps_predict(&rps_selftest_frames[k * 64], &result);
        for (int j = 0; j < 3; ++j) {
            float delta = fabsf(result.probabilities[j] - rps_selftest_probabilities[k * 3 + j]);
            if (!isfinite(delta)) { *max_probability_error = INFINITY; return false; }
            error = fmaxf(error, delta);
        }
    }
    *max_probability_error = error;
    return error < 0.0001f;
}
