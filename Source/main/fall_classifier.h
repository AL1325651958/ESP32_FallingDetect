#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifndef FALL_USE_INT8
#define FALL_USE_INT8 0
#endif

#define FALL_WINDOW_FRAMES 24
#define FALL_ZONES 64

enum {
    FALL_UNKNOWN = -1, FALL_MOTION = 0, FALL_DETECTED = 1, FALL_PERSON_STILL = 2,
    FALL_OBSERVATION_LOST = 3, FALL_SUSPECTED = 4
};

typedef struct {
    int16_t frames[FALL_WINDOW_FRAMES][FALL_ZONES];
    uint32_t last_ms;
    uint8_t count, next, high_count, low_count;
    uint8_t predict_phase;
    int8_t stable, raw;
    bool has_time;
    float probabilities[3];
    uint32_t motion_history, downward_history;
    uint32_t last_motion_ms, last_downward_ms, last_candidate_ms;
    bool has_motion, has_downward, has_candidate, observation_lost;
    int8_t held_alert;
} fall_state_t;

typedef struct {
    int8_t raw, stable; /* 3: observation lost, never proof of vacancy; 4: suspected fall */
    uint8_t valid_zones, buffered_frames;
    float probability[3]; /* motion, fall, person-still; uncalibrated scores */
    uint32_t inference_us;
    bool prediction_updated;
} fall_result_t;

void fall_reset(fall_state_t *state);
void fall_push(fall_state_t *state, const int16_t distances[64], uint32_t now_ms, fall_result_t *result);
float fall_predict_window(const int16_t distances[FALL_WINDOW_FRAMES][FALL_ZONES], float probabilities[3]);
float fall_predict_sequence(const int16_t *distances, unsigned frames, float probabilities[3]);
bool fall_self_test(float *max_error);
