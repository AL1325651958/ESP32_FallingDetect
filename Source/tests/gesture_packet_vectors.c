#include <stdio.h>
#include "rps_telemetry.h"

int main(void)
{
    const float scores[3][3] = {{0.8f,0.15f,0.05f},{0.1f,0.85f,0.05f},{0.01f,0.04f,0.95f}};
    const uint16_t frames[] = {0,255,65535};
    for (int k = 0; k < 3; ++k) {
        rps_result_t result = {.raw_label = (rps_label_t)k, .valid_zones = 64, .near_zones = 12};
        memcpy(result.probabilities, scores[k], sizeof(scores[k]));
        uint8_t payload[RPS_TELEMETRY_BYTES];
        rps_telemetry_encode(payload, frames[k], &result, k == 0 ? RPS_UNKNOWN : (rps_label_t)k);
        for (int i=0; i<RPS_TELEMETRY_BYTES; ++i) printf("%02X", payload[i]);
        putchar('\n');
    }
    return 0;
}
