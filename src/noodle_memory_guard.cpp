#include "noodle_memory_guard.h"
#include <limits.h>

#if NOODLE_MEMORY_GUARD && defined(__AVR__)
#include <avr/io.h>

extern "C" {
extern char __heap_start;
extern char *__brkval;
}

static volatile int16_t guard_min_gap = INT16_MAX;
static volatile int16_t guard_last_gap = INT16_MAX;
static volatile uint16_t guard_samples = 0;
static volatile uint8_t guard_collision = 0;

void noodle_memory_guard_reset(void) {
  guard_min_gap = INT16_MAX;
  guard_last_gap = INT16_MAX;
  guard_samples = 0;
  guard_collision = 0;
}

/* noinline lets the sample include this function's small stack frame.  AVR's
 * SP register is readable on ATmega328P and ATmega2560.  Sampling may still
 * miss deeper transient stack use within SdFat/interrupt handlers. */
__attribute__((noinline)) void noodle_memory_guard_checkpoint(void) {
  const uintptr_t heap = __brkval ? (uintptr_t)__brkval : (uintptr_t)&__heap_start;
  const uintptr_t sp = (uintptr_t)SP;
  const int16_t gap = (int16_t)((int32_t)sp - (int32_t)heap);
  guard_last_gap = gap;
  if (gap < guard_min_gap) guard_min_gap = gap;
  if (gap <= 0) guard_collision = 1;
  if (guard_samples != UINT16_MAX) ++guard_samples;
}

NoodleMemoryGuardReport noodle_memory_guard_report(void) {
  NoodleMemoryGuardReport r;
  /* Snapshot outside an ISR.  This function should be called from the normal
   * sketch loop after inference; a concurrent ISR does not update the guard. */
  r.current_gap_bytes = (guard_last_gap == INT16_MAX) ? 0 : guard_last_gap;
  r.minimum_gap_bytes = (guard_min_gap == INT16_MAX) ? 0 : guard_min_gap;
  r.sample_count = guard_samples;
  r.observed_collision = guard_collision;
  r.supported = 1;
  return r;
}
#else
void noodle_memory_guard_reset(void) {}
void noodle_memory_guard_checkpoint(void) {}
NoodleMemoryGuardReport noodle_memory_guard_report(void) {
  NoodleMemoryGuardReport r = {0, 0, 0, 0, 0};
  return r;
}
#endif

uint16_t noodle_memory_guard_checksum_i8(const int8_t *data, uint16_t count) {
  if (!data) return 0;
  uint16_t hash = 0xA5A5u;
  for (uint16_t i = 0; i < count; ++i) {
    hash = (uint16_t)((hash << 5) | (hash >> 11));
    hash ^= (uint8_t)data[i];
    hash = (uint16_t)(hash + 0x9E37u + i);
  }
  return hash;
}
