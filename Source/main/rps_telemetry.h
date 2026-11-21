#pragma once
#include <stdint.h>
#include <string.h>
#include "rps_classifier.h"

#define RPS_TELEMETRY_BYTES 20
#define RPS_TELEMETRY_MAGIC 0xA7
#define RPS_TELEMETRY_VERSION 1

/* Default ATT MTU payload: A7, version, frame LE16, raw/stable int8,
 * valid/near uint8, then rock/paper/scissors IEEE754 float32 little-endian.
 * Stable -1 (wire FF) means rejected or fewer than three matching frames. */
static inline void rps_telemetry_encode(uint8_t payload[RPS_TELEMETRY_BYTES],
                                        uint16_t frame, const rps_result_t *prediction,
                                        rps_label_t stable)
{
    _Static_assert(sizeof(float) == 4, "Telemetry requires float32");
    payload[0] = RPS_TELEMETRY_MAGIC;
    payload[1] = RPS_TELEMETRY_VERSION;
    payload[2] = (uint8_t)frame;
    payload[3] = (uint8_t)(frame >> 8);
    payload[4] = (uint8_t)(int8_t)prediction->raw_label;
    payload[5] = (uint8_t)(int8_t)stable;
    payload[6] = prediction->valid_zones;
    payload[7] = prediction->near_zones;
    for (int k = 0; k < RPS_CLASS_COUNT; ++k) {
        uint32_t bits;
        memcpy(&bits, &prediction->probabilities[k], sizeof(bits));
        for (int b = 0; b < 4; ++b)
            payload[8 + k * 4 + b] = (uint8_t)(bits >> (b * 8));
    }
}
