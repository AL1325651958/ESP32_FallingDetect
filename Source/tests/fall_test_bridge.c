#include "fall_telemetry.h"
#include <stddef.h>
unsigned fall_state_size(void) { return sizeof(fall_state_t); }
unsigned fall_result_size(void) { return sizeof(fall_result_t); }
void fall_encode_for_test(uint8_t *packet, uint16_t frame, const fall_result_t *result)
{
    fall_telemetry_encode(packet, frame, result);
}
