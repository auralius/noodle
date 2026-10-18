#ifndef NOODLE_MEMORY_GUARD_H
#define NOODLE_MEMORY_GUARD_H

#include <stdint.h>
#include <stddef.h>

#ifndef NOODLE_MEMORY_GUARD
#define NOODLE_MEMORY_GUARD 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* AVR diagnostic only: every sample records the *instantaneous* stack-to-heap
 * distance, not a verified worst-case peak.  No memory allocation or I/O.
 * Compile with -DNOODLE_MEMORY_GUARD=1 for diagnosis; disable for timing.
 */
typedef struct {
  int16_t current_gap_bytes;   /* negative means heap/stack overlap at sample */
  int16_t minimum_gap_bytes;   /* minimum observed among sampled checkpoints */
  uint16_t sample_count;
  uint8_t observed_collision;
  uint8_t supported;          /* 1 on AVR, 0 on other architectures */
} NoodleMemoryGuardReport;

void noodle_memory_guard_reset(void);
void noodle_memory_guard_checkpoint(void);
NoodleMemoryGuardReport noodle_memory_guard_report(void);

/* Allocation-free checksum for comparing read-only activation snapshots.
 * Equality is an integrity screening check, not a cryptographic guarantee.
 */
uint16_t noodle_memory_guard_checksum_i8(const int8_t *buffer, uint16_t count);

#ifdef __cplusplus
}
#endif
#endif
