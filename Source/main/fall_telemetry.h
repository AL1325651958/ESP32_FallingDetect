#pragma once
#include "fall_classifier.h"
#include <string.h>

#define FALL_TELEMETRY_BYTES 20

static inline void fall_write_float_le(uint8_t *p, float value)
{
    uint32_t bits;
    _Static_assert(sizeof(float) == 4, "32-bit IEEE754 float required");
    memcpy(&bits, &value, 4);
    for (unsigned i = 0; i < 4; ++i) p[i] = (uint8_t)(bits >> (8 * i));
}

static inline void fall_telemetry_encode(uint8_t payload[20], uint16_t frame, const fall_result_t *r)
{
    payload[0] = 0xA8; payload[1] = 2;
    payload[2] = (uint8_t)frame; payload[3] = (uint8_t)(frame >> 8);
    payload[4] = (uint8_t)r->raw; payload[5] = (uint8_t)r->stable;
    payload[6] = r->valid_zones; payload[7] = r->buffered_frames;
    fall_write_float_le(payload + 8, r->probability[0]);
    fall_write_float_le(payload + 12, r->probability[1]);
    for (unsigned i = 0; i < 4; ++i) payload[16 + i] = (uint8_t)(r->inference_us >> (8 * i));
}
