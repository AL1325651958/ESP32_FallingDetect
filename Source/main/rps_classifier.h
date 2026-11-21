#pragma once
#include <stdbool.h>
#include <stdint.h>

#define RPS_CLASS_COUNT 3
#define RPS_ZONE_COUNT 64
#define RPS_CONFIDENCE_THRESHOLD 0.75f

typedef enum { RPS_UNKNOWN = -1, RPS_ROCK = 0, RPS_PAPER = 1, RPS_SCISSORS = 2 } rps_label_t;
typedef struct {
    rps_label_t raw_label;
    rps_label_t label;
    float probabilities[RPS_CLASS_COUNT];
    float confidence;
    uint8_t valid_zones;
    uint8_t near_zones;
} rps_result_t;

/* Sensor row-major order. -1 and UINT16_MAX are invalid.
 * Uses static scratch buffers: call from ONE task (not reentrant).
 * Does not allocate memory or use PSRAM. */
void rps_predict(const int16_t distances_mm[64], rps_result_t *result);
const char *rps_label_name(rps_label_t label);
/* Compare C output to Python reference probabilities on three held-out frames. */
bool rps_self_test(float *max_probability_error);
