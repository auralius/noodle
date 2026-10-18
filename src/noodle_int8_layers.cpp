#include "noodle_config.h"
#if defined(NOODLE_USE_INT8)
#include "noodle_int8.h"
#include "noodle_internal.h"
#include "noodle_memory_guard.h"
#if defined(__AVR__) && NOODLE_MEMORY_GUARD
#define NDL_MEM_CHECK() noodle_memory_guard_checkpoint()
#else
#define NDL_MEM_CHECK() ((void)0)
#endif

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// AVR FCN dispatch keeps generic and file-input kernels in independent
// call frames. This intentionally limits the active stack footprint on
// memory-constrained AVR boards; it is not a user-selectable build option.

// Temporary AVR-only FCN progress trace. Compiled away unless explicitly enabled.
// Does not change tensor placement or parameter I/O order.
#if defined(__AVR__) && defined(NOODLE_INT8_FCN_PROGRESS) && NOODLE_INT8_FCN_PROGRESS
#include <Arduino.h>
#define NDL_FCN_MARK(what, neuron) do { \
  Serial.print(F("DBG NDL ")); \
  Serial.write((char)(what)); \
  Serial.write(' '); \
  Serial.println((unsigned int)(neuron)); \
} while (0)
#else
#define NDL_FCN_MARK(what, neuron) ((void)0)
#endif

// Optional, 1-byte INT8 FCN failure trace for memory-constrained boards.
// Enabled only for builds with -D NOODLE_INT8_FCN_DEBUG=1.
#if defined(NOODLE_INT8_FCN_DEBUG) && NOODLE_INT8_FCN_DEBUG
static volatile uint8_t g_noodle_i8_fcn_debug_stage = 0;
uint8_t noodle_i8_fcn_debug_stage() { return g_noodle_i8_fcn_debug_stage; }
#define NDL_FCN_STAGE(stage) (g_noodle_i8_fcn_debug_stage = (stage))
#else
#define NDL_FCN_STAGE(stage) ((void)0)
#endif

#if NOODLE_INT8_OUTPUT_TILE < 1
#error "NOODLE_INT8_OUTPUT_TILE must be at least 1"
#endif
#if NOODLE_INT8_PW_OUTPUT_BLOCK < 1
#error "NOODLE_INT8_PW_OUTPUT_BLOCK must be at least 1"
#endif

static int32_t *g_q_acc = nullptr;
static size_t g_q_acc_capacity = 0;
static int8_t *g_q_filter = nullptr;
static size_t g_q_filter_capacity = 0;

#if defined(ARDUINO_ARCH_ESP32)
#include <Arduino.h>
#endif

static void *q_alloc(size_t bytes) {
  if (!bytes) return nullptr;
  void *p = malloc(bytes);
#if defined(ARDUINO_ARCH_ESP32)
  if (!p && psramFound()) p = ps_malloc(bytes);
#endif
  return p;
}

static int32_t *q_acc_require(size_t n) {
  if (n <= g_q_acc_capacity) return g_q_acc;
  if (n > static_cast<size_t>(-1) / sizeof(int32_t)) return nullptr;
  int32_t *p = static_cast<int32_t *>(q_alloc(n * sizeof(int32_t)));
  if (!p) return nullptr;
  free(g_q_acc);
  g_q_acc = p;
  g_q_acc_capacity = n;
  return p;
}

static int8_t *q_filter_require(size_t n) {
  if (n <= g_q_filter_capacity) return g_q_filter;
  int8_t *p = static_cast<int8_t *>(q_alloc(n));
  if (!p) return nullptr;
  free(g_q_filter);
  g_q_filter = p;
  g_q_filter_capacity = n;
  return p;
}

void noodle_i8_temp_buffers_free(void) {
  free(g_q_acc);
  free(g_q_filter);
  g_q_acc = nullptr;
  g_q_filter = nullptr;
  g_q_acc_capacity = 0;
  g_q_filter_capacity = 0;
}

static int32_t q_sat_i32(int64_t x) {
  if (x > INT32_MAX) return INT32_MAX;
  if (x < INT32_MIN) return INT32_MIN;
  return static_cast<int32_t>(x);
}

static bool q_scale_equal(float a, float b) {
  const float d = fabsf(a - b);
  const float m = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
  return d <= 1e-6f * m;
}

static bool q_input_matches(const NoodleTensor *input,
                            float scale, int32_t zp) {
  return input && q_scale_equal(input->scale, scale) && input->zero_point == zp;
}

static int32_t q_apply_output(int32_t acc, int32_t multiplier, int32_t shift,
                              int32_t output_zp, int32_t act_min,
                              int32_t act_max) {
  int32_t q = noodle_i8_multiply_by_quantized_multiplier(acc, multiplier, shift);
  int64_t with_zp = static_cast<int64_t>(q) + output_zp;
  q = q_sat_i32(with_zp);
  if (q < act_min) q = act_min;
  if (q > act_max) q = act_max;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return q;
}

struct QGeom {
  uint16_t V;
  uint16_t P0;
  uint16_t P1;
};

static bool q_conv_geom(uint16_t W, uint16_t K, uint16_t P, uint16_t S,
                        QGeom *g) {
  if (!g || !W || !K || !S) return false;
  if (P == NOODLE_SAME_PADDING) {
    const uint32_t V = (static_cast<uint32_t>(W) + S - 1u) / S;
    const uint32_t needed = V ? (V - 1u) * S + K : 0;
    const uint32_t total = needed > W ? needed - W : 0;
    if (V > 65535u || total > 131070u) return false;
    g->V = static_cast<uint16_t>(V);
    g->P0 = static_cast<uint16_t>(total / 2u);
    g->P1 = static_cast<uint16_t>(total - total / 2u);
    return true;
  }
  const int32_t num = static_cast<int32_t>(W) - K + 2 * static_cast<int32_t>(P);
  if (num < 0) return false;
  g->V = static_cast<uint16_t>(num / S + 1);
  g->P0 = g->P1 = P;
  return g->V != 0;
}

static size_t q_read_i8_block(NDL_File &f, int8_t *dst, size_t n) {
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  return noodle_read_raw(f, dst, n);
#else
  for (size_t i = 0; i < n; ++i) dst[i] = noodle_read_q8(f);
  return n;
#endif
}

static bool q_open_required(const char *fn, NDL_File *f) {
  if (!fn || !f) return false;
  *f = noodle_fs_open_read(fn);
  return static_cast<bool>(*f);
}

static void q_close_if(NDL_File *f) {
  if (f && static_cast<bool>(*f)) f->close();
}

static int32_t q_conv2d_at(const int8_t *input, uint16_t I, uint16_t W,
                           const int8_t *filter, uint16_t K,
                           uint16_t oy, uint16_t ox,
                           uint16_t P0, uint16_t S,
                           int32_t input_zp, int32_t bias) {
  int64_t acc = bias;
  const uint32_t in_plane = static_cast<uint32_t>(W) * W;
  const uint32_t kernel_plane = static_cast<uint32_t>(K) * K;
  for (uint16_t ic = 0; ic < I; ++ic) {
    const int8_t *x = input + static_cast<uint32_t>(ic) * in_plane;
    const int8_t *w = filter + static_cast<uint32_t>(ic) * kernel_plane;
    for (uint16_t ky = 0; ky < K; ++ky) {
      const int32_t iy = static_cast<int32_t>(oy) * S + ky - P0;
      if (iy < 0 || iy >= W) continue;
      for (uint16_t kx = 0; kx < K; ++kx) {
        const int32_t ix = static_cast<int32_t>(ox) * S + kx - P0;
        if (ix < 0 || ix >= W) continue;
        const int32_t xv = static_cast<int32_t>(x[static_cast<uint32_t>(iy) * W + ix]) - input_zp;
        acc += static_cast<int64_t>(xv) * w[static_cast<uint32_t>(ky) * K + kx];
      }
    }
  }
  return q_sat_i32(acc);
}

static bool q_conv2d_filter_direct(const NoodleTensor *input, uint16_t V,
                                   uint16_t P0, const int8_t *filter,
                                   int32_t bias, int32_t multiplier,
                                   int32_t shift, uint16_t K, uint16_t S,
                                   int32_t input_zp, int32_t output_zp,
                                   int32_t act_min, int32_t act_max,
                                   int8_t *out_plane) {
  if (!input || !filter || !out_plane) return false;
  const int8_t *src = input->buffer.data;
  for (uint16_t oy = 0; oy < V; ++oy)
    for (uint16_t ox = 0; ox < V; ++ox) {
      const int32_t acc = q_conv2d_at(src, input->C, input->W, filter, K,
                                      oy, ox, P0, S, input_zp, bias);
      out_plane[static_cast<uint32_t>(oy) * V + ox] = static_cast<int8_t>(
          q_apply_output(acc, multiplier, shift, output_zp, act_min, act_max));
    }
  return true;
}

// Spatially tiled INT8 3x3 convolution.  The filter is still supplied by the
// caller, so file-backed layers can keep streaming one output filter at a
// time.  Compared with q_conv2d_filter_direct(), this walks input rows and
// output tiles explicitly, avoiding repeated q_conv2d_at() geometry work.
static bool q_conv2d_filter_streaming_3x3(
    const NoodleTensor *input, uint16_t V, uint16_t P0,
    const int8_t *filter, int32_t bias, int32_t multiplier, int32_t shift,
    uint16_t S, int32_t input_zp, int32_t output_zp,
    int32_t act_min, int32_t act_max, int8_t *out_plane) {
  if (!input || !filter || !out_plane || input->W == 0 || V == 0 || S == 0)
    return false;

  const uint16_t W = input->W;
  const uint16_t tile = static_cast<uint16_t>(NOODLE_INT8_OUTPUT_TILE);
  int64_t tile_acc[NOODLE_INT8_OUTPUT_TILE * NOODLE_INT8_OUTPUT_TILE];
  const uint32_t in_plane = static_cast<uint32_t>(W) * W;

  for (uint16_t oy0 = 0; oy0 < V; oy0 = static_cast<uint16_t>(oy0 + tile)) {
    const uint16_t oy1 = (oy0 + tile < V) ? static_cast<uint16_t>(oy0 + tile) : V;
    for (uint16_t ox0 = 0; ox0 < V; ox0 = static_cast<uint16_t>(ox0 + tile)) {
      const uint16_t ox1 = (ox0 + tile < V) ? static_cast<uint16_t>(ox0 + tile) : V;

      for (uint16_t oy = oy0; oy < oy1; ++oy)
        for (uint16_t ox = ox0; ox < ox1; ++ox)
          tile_acc[static_cast<size_t>(oy - oy0) * tile + (ox - ox0)] = bias;

      for (uint16_t ic = 0; ic < input->C; ++ic) {
        const int8_t *x = input->buffer.data + static_cast<uint32_t>(ic) * in_plane;
        const int8_t *w = filter + static_cast<uint32_t>(ic) * 9u;

        for (uint16_t oy = oy0; oy < oy1; ++oy) {
          const int32_t iy0 = static_cast<int32_t>(oy) * S - P0;
          for (uint16_t ky = 0; ky < 3; ++ky) {
            const int32_t iy = iy0 + ky;
            if (iy < 0 || iy >= W) continue;
            const int8_t *row = x + static_cast<uint32_t>(iy) * W;
            const int8_t *wk = w + static_cast<uint32_t>(ky) * 3u;

            for (uint16_t ox = ox0; ox < ox1; ++ox) {
              const int32_t ix0 = static_cast<int32_t>(ox) * S - P0;
              int32_t kx0 = ix0 < 0 ? -ix0 : 0;
              int32_t kx1 = W - ix0;
              if (kx1 > 3) kx1 = 3;
              if (kx0 >= kx1) continue;

              int64_t &a = tile_acc[static_cast<size_t>(oy - oy0) * tile + (ox - ox0)];
              for (int32_t kx = kx0; kx < kx1; ++kx) {
                const int32_t xv = static_cast<int32_t>(row[ix0 + kx]) - input_zp;
                a += static_cast<int64_t>(xv) * wk[kx];
              }
            }
          }
        }
      }

      for (uint16_t oy = oy0; oy < oy1; ++oy)
        for (uint16_t ox = ox0; ox < ox1; ++ox) {
          const int64_t a = tile_acc[static_cast<size_t>(oy - oy0) * tile + (ox - ox0)];
          out_plane[static_cast<uint32_t>(oy) * V + ox] = static_cast<int8_t>(
              q_apply_output(q_sat_i32(a), multiplier, shift, output_zp,
                             act_min, act_max));
        }
    }
  }
  return true;
}

static bool q_conv2d_filter(const NoodleTensor *input, uint16_t V,
                            uint16_t P0, const int8_t *filter,
                            int32_t bias, int32_t multiplier, int32_t shift,
                            uint16_t K, uint16_t S,
                            int32_t input_zp, int32_t output_zp,
                            int32_t act_min, int32_t act_max,
                            int8_t *out_plane) {
#if NOODLE_INT8_CONV_IMPL == NOODLE_INT8_CONV_IMPL_STREAMING
  if (K == 3 && q_conv2d_filter_streaming_3x3(
                   input, V, P0, filter, bias, multiplier, shift, S,
                   input_zp, output_zp, act_min, act_max, out_plane))
    return true;
#endif
  return q_conv2d_filter_direct(input, V, P0, filter, bias, multiplier,
                                shift, K, S, input_zp, output_zp, act_min,
                                act_max, out_plane);
}

// Blocked INT8 pointwise (1x1) convolution.  Filters are packed as
// [block_output_channel][input_channel].  A small output-channel block reuses
// each input value across several filters before moving to the next tile.
static bool q_conv2d_pw_blocked(
    const NoodleTensor *input, uint16_t V, uint16_t P0, uint16_t S,
    const int8_t *filter_block, uint16_t block_o,
    const int32_t *bias_block, const int32_t *multiplier_block,
    const int32_t *shift_block, int32_t input_zp, int32_t output_zp,
    int32_t act_min, int32_t act_max, int8_t *dst_block) {
  if (!input || !input->buffer.data || !filter_block || !block_o ||
      !bias_block || !multiplier_block || !shift_block || !dst_block ||
      !V || !S) return false;

  const uint16_t W = input->W;
  const uint16_t tile = static_cast<uint16_t>(NOODLE_INT8_OUTPUT_TILE);
  const size_t tile_plane = static_cast<size_t>(tile) * tile;
  const size_t tile_acc_count = static_cast<size_t>(block_o) * tile_plane;
  if (tile_acc_count > static_cast<size_t>(-1) / sizeof(int64_t)) return false;
  // This scratch is deliberately per invocation.  Hybrid branches execute
  // concurrently on the ESP32 dual-core runner; a shared global accumulator
  // lets the two branches overwrite one another and corrupt detections.
  int64_t *tile_acc = static_cast<int64_t *>(
      q_alloc(tile_acc_count * sizeof(int64_t)));
  if (!tile_acc) return false;

  const uint32_t in_plane = static_cast<uint32_t>(W) * W;
  const uint32_t out_plane = static_cast<uint32_t>(V) * V;
  const uint16_t I = input->C;

  for (uint16_t oy0 = 0; oy0 < V; oy0 = static_cast<uint16_t>(oy0 + tile)) {
    const uint16_t oy1 = (oy0 + tile < V) ? static_cast<uint16_t>(oy0 + tile) : V;
    for (uint16_t ox0 = 0; ox0 < V; ox0 = static_cast<uint16_t>(ox0 + tile)) {
      const uint16_t ox1 = (ox0 + tile < V) ? static_cast<uint16_t>(ox0 + tile) : V;

      for (uint16_t bo = 0; bo < block_o; ++bo)
        for (uint16_t oy = oy0; oy < oy1; ++oy)
          for (uint16_t ox = ox0; ox < ox1; ++ox)
            tile_acc[static_cast<size_t>(bo) * tile_plane +
                     static_cast<size_t>(oy - oy0) * tile + (ox - ox0)] =
                bias_block[bo];

      for (uint16_t ic = 0; ic < I; ++ic) {
        const int8_t *x = input->buffer.data + static_cast<uint32_t>(ic) * in_plane;
        for (uint16_t oy = oy0; oy < oy1; ++oy) {
          const int32_t iy = static_cast<int32_t>(oy) * S - P0;
          if (iy < 0 || iy >= W) continue;
          const int8_t *row = x + static_cast<uint32_t>(iy) * W;
          for (uint16_t ox = ox0; ox < ox1; ++ox) {
            const int32_t ix = static_cast<int32_t>(ox) * S - P0;
            if (ix < 0 || ix >= W) continue;
            const int32_t xv = static_cast<int32_t>(row[ix]) - input_zp;
            const size_t ai = static_cast<size_t>(oy - oy0) * tile + (ox - ox0);
            for (uint16_t bo = 0; bo < block_o; ++bo) {
              tile_acc[static_cast<size_t>(bo) * tile_plane + ai] +=
                  static_cast<int64_t>(xv) *
                  filter_block[static_cast<size_t>(bo) * I + ic];
            }
          }
        }
      }

      for (uint16_t bo = 0; bo < block_o; ++bo) {
        int8_t *out = dst_block + static_cast<uint32_t>(bo) * out_plane;
        for (uint16_t oy = oy0; oy < oy1; ++oy)
          for (uint16_t ox = ox0; ox < ox1; ++ox) {
            const size_t ai = static_cast<size_t>(bo) * tile_plane +
                              static_cast<size_t>(oy - oy0) * tile + (ox - ox0);
            out[static_cast<uint32_t>(oy) * V + ox] = static_cast<int8_t>(
                q_apply_output(q_sat_i32(tile_acc[ai]), multiplier_block[bo],
                               shift_block[bo], output_zp, act_min, act_max));
          }
      }
    }
  }
  free(tile_acc);
  return true;
}


// Optional AVR fused Conv2D / DepthwiseConv2D 32-bit dot accumulation.
// Worst-case |(qinput-zp)*weight| <= 255*128 = 32640 when zp is int8.
// Keep the INT64 fallback for networks that do not satisfy this bound.
#ifndef NOODLE_INT8_CONV_ACC32
#define NOODLE_INT8_CONV_ACC32 0
#endif
#if defined(__AVR__) && NOODLE_INT8_CONV_ACC32
#pragma message("Noodle fused INT8 Conv/DWConv ACC32 enabled")
#endif
// For non-AVR MCUs, choose INT32 for bounded INT8 dot products without
// requiring a build flag. AVR keeps its original Case-1 opt-in behavior.
// Accumulation starts at zero; bias is always added later in INT64 and then
// saturated to INT32, exactly as in the pre-existing INT64 implementation.
static bool q_fused_acc32_safe(uint64_t products, int32_t input_zp) {
#if !defined(__AVR__) || NOODLE_INT8_CONV_ACC32
  return input_zp >= -128 && input_zp <= 127 &&
         products <= static_cast<uint64_t>(INT32_MAX) / (255u * 128u);
#else
  (void)products;
  (void)input_zp;
  return false;
#endif
}

// Optional SD-backed fused pointwise (1x1, stride 1, identity-pool) row cache.
// Holds only C * W input bytes, rather than a complete C * W * W tensor.
// Independent of NOODLE_INT8_CONV_ACC32 and non-fused PW/CONV flags.
#ifndef NOODLE_INT8_SD_POINTWISE_ROWBUF
#define NOODLE_INT8_SD_POINTWISE_ROWBUF 0
#endif
#if NOODLE_INT8_SD_POINTWISE_ROWBUF
#pragma message("Noodle fused INT8 SD pointwise row buffer enabled")
#endif

// Optional general K-row SD activation cache for fused Conv2D/DWConv2D.
// K controls the sliding window (not a hard-coded 3x3 kernel). Read CHW rows
// once and retain them for overlapping output positions. Input/output stay on SD.
// For regular Conv2D, cache K rows PER input channel; for depthwise Conv2D,
// keep just one channel's K rows, invalidating on channel changes.
#ifndef NOODLE_INT8_SD_KROW_CACHE
#define NOODLE_INT8_SD_KROW_CACHE 0
#endif
#ifndef NOODLE_INT8_SD_KROW_MAX_BYTES
#define NOODLE_INT8_SD_KROW_MAX_BYTES 512
#endif
#if NOODLE_INT8_SD_KROW_CACHE
#pragma message("Noodle fused INT8 SD general K-row cache enabled")
#endif

// NOODLE_INT8_SD_PLANE_FAST_PATCH_V1: general file-backed input-plane prefetch for Conv/DWConv.
// DWConv keeps one C-channel plane in SRAM; ordinary Conv2D retains ALL
// input channels only when they fit the configured budget. Low-memory
// fallback uses the existing K-row (or single-row) reader.
#ifndef NOODLE_INT8_SD_PLANE_FAST
#define NOODLE_INT8_SD_PLANE_FAST 0
#endif
#ifndef NOODLE_INT8_SD_PLANE_MAX_BYTES
#define NOODLE_INT8_SD_PLANE_MAX_BYTES 512
#endif
#if NOODLE_INT8_SD_PLANE_FAST
#pragma message("Noodle fused INT8 SD input-plane prefetch enabled")
#endif

// ----- Fused INT8 convolution -> pooling, including file-backed activations -----
// No full pre-pool convolution tensor is ever allocated. File-backed inputs
// use a single CHW row cache, while file-backed outputs are streamed in CHW
// order. Other kernels preserve their existing optimized SRAM-only fast paths.
struct QI8ActivationReader {
  const NoodleTensor *tensor;
  NDL_File file;
  int8_t *row;
  int32_t cached_channel;
  int32_t cached_y;
  bool valid;
#if NOODLE_INT8_SD_PLANE_FAST
  int8_t *plane_data;
  size_t plane_stride;
  uint16_t plane_channel;
  bool plane_single_channel;
#endif
#if NOODLE_INT8_SD_KROW_CACHE
  int8_t *window_rows;
  uint16_t *window_tags;
  uint16_t window_k;
  uint16_t window_channels;
  uint16_t window_current_channel;
  bool window_single_channel;
#endif
  explicit QI8ActivationReader(const NoodleTensor *t)
      : tensor(t), row(nullptr), cached_channel(-1), cached_y(-1), valid(true)
#if NOODLE_INT8_SD_PLANE_FAST
      , plane_data(nullptr), plane_stride(0), plane_channel(UINT16_MAX),
        plane_single_channel(false)
#endif
#if NOODLE_INT8_SD_KROW_CACHE
      , window_rows(nullptr), window_tags(nullptr), window_k(0),
        window_channels(0), window_current_channel(UINT16_MAX),
        window_single_channel(false)
#endif
  {
    if (!t || (!t->buffer.data && !t->file_path)) { valid = false; return; }
    if (!t->file_path) return;
    file = noodle_fs_open_read(t->file_path);
    const uint64_t size = static_cast<uint64_t>(t->C) * t->W * t->W;
    if (!file || file.size() != size) { valid = false; return; }
    row = static_cast<int8_t *>(q_alloc(t->W));
    if (!row) valid = false;
  }
  ~QI8ActivationReader() {
    if (file) file.close();
    free(row);
#if NOODLE_INT8_SD_PLANE_FAST
    free(plane_data);
#endif
#if NOODLE_INT8_SD_KROW_CACHE
    free(window_rows);
    free(window_tags);
#endif
  }
#if NOODLE_INT8_SD_KROW_CACHE
  // Opportunistic cache: failure or a configured size limit leaves the
  // original one-row reader active (no inference failure).
  bool enable_krow(uint16_t K, bool single_channel) {
    if (!valid || !tensor || !tensor->file_path || K <= 1 || window_rows)
      return false;
    const uint32_t channels = single_channel ? 1u : tensor->C;
    const uint64_t slots = static_cast<uint64_t>(channels) * K;
    const uint64_t row_bytes = slots * tensor->W;
    const uint64_t tag_bytes = slots * sizeof(uint16_t);
    const uint64_t budget = static_cast<uint64_t>(NOODLE_INT8_SD_KROW_MAX_BYTES);
    if (!slots || !row_bytes || row_bytes + tag_bytes > budget ||
        row_bytes > static_cast<size_t>(-1) ||
        tag_bytes > static_cast<size_t>(-1)) return false;
    int8_t *new_rows = static_cast<int8_t *>(q_alloc(static_cast<size_t>(row_bytes)));
    if (!new_rows) return false;
    uint16_t *new_tags = static_cast<uint16_t *>(q_alloc(static_cast<size_t>(tag_bytes)));
    if (!new_tags) { free(new_rows); return false; }
    memset(new_tags, 0xff, static_cast<size_t>(tag_bytes));
    window_rows = new_rows;
    window_tags = new_tags;
    window_k = K;
    window_channels = static_cast<uint16_t>(channels);
    window_current_channel = UINT16_MAX;
    window_single_channel = single_channel;
    return true;
  }
#endif
#if NOODLE_INT8_SD_PLANE_FAST
  // Failure to allocate falls back to normal SD reader. Failed SD reads
  // invalidate the reader and must NOT be mistaken for allocation fallback.
  bool enable_plane(bool single_channel) {
    if (!valid || !tensor || !tensor->file_path || plane_data) return false;
    const uint64_t plane = static_cast<uint64_t>(tensor->W) * tensor->W;
    const uint64_t bytes = plane * (single_channel ? 1u : tensor->C);
    if (!plane || !bytes || bytes > NOODLE_INT8_SD_PLANE_MAX_BYTES ||
        bytes > static_cast<size_t>(-1)) return false;
    int8_t *p = static_cast<int8_t *>(q_alloc(static_cast<size_t>(bytes)));
    if (!p) return false;
    if (!single_channel) {
      if (!file.seek(0) || noodle_read_raw(file, p, static_cast<size_t>(bytes)) != bytes) {
        free(p); valid = false; return false;
      }
    }
    plane_data = p;
    plane_stride = static_cast<size_t>(plane);
    plane_channel = UINT16_MAX;
    plane_single_channel = single_channel;
    return true;
  }
  const int8_t *channel_plane(uint16_t c) {
    if (!valid || !plane_data || c >= tensor->C) return nullptr;
    if (!plane_single_channel)
      return plane_data + static_cast<size_t>(c) * plane_stride;
    if (plane_channel != c) {
      const uint64_t offset = static_cast<uint64_t>(c) * plane_stride;
      if (offset > UINT32_MAX ||
          !file.seek(static_cast<uint32_t>(offset)) ||
          noodle_read_raw(file, plane_data, plane_stride) != plane_stride) {
        valid = false;
        return nullptr;
      }
      plane_channel = c;
    }
    return plane_data;
  }
#endif
  bool at(uint16_t c, uint16_t y, uint16_t x, int8_t *v) {
    if (!valid || !v || c >= tensor->C || y >= tensor->W || x >= tensor->W)
      return false;
    if (!tensor->file_path) {
      *v = tensor->buffer.data[(static_cast<size_t>(c) * tensor->W + y) * tensor->W + x];
      return true;
    }
#if NOODLE_INT8_SD_PLANE_FAST
    if (plane_data) {
      const int8_t *p = channel_plane(c);
      if (!p) return false;
      *v = p[static_cast<size_t>(y) * tensor->W + x];
      return true;
    }
#endif
#if NOODLE_INT8_SD_KROW_CACHE
    if (window_rows) {
      if (window_single_channel && c != window_current_channel) {
        // Depthwise layers visit all pixels of a channel before changing c.
        memset(window_tags, 0xff,
               static_cast<size_t>(window_channels) * window_k * sizeof(uint16_t));
        window_current_channel = c;
      }
      const size_t slot = (static_cast<size_t>(window_single_channel ? 0u : c) *
                           window_k) + (y % window_k);
      int8_t *r = window_rows + slot * tensor->W;
      if (window_tags[slot] != y) {
        const uint64_t offset = (static_cast<uint64_t>(c) * tensor->W + y) * tensor->W;
        if (offset > UINT32_MAX || !file.seek(static_cast<uint32_t>(offset)) ||
            noodle_read_raw(file, r, tensor->W) != tensor->W) {
          valid = false;
          return false;
        }
        window_tags[slot] = y;
      }
      *v = r[x];
      return true;
    }
#endif
    if (cached_channel != c || cached_y != y) {
      const uint64_t offset = (static_cast<uint64_t>(c) * tensor->W + y) * tensor->W;
      if (offset > UINT32_MAX || !file.seek(static_cast<uint32_t>(offset)) ||
          noodle_read_raw(file, row, tensor->W) != tensor->W) {
        valid = false;
        return false;
      }
      cached_channel = c;
      cached_y = y;
    }
    *v = row[x];
    return true;
  }
#if NOODLE_INT8_SD_POINTWISE_ROWBUF
  // Copy one CHW row in one SD access (or one memcpy from SRAM).
  // Reuses the existing single-row cache; never materializes a feature map.
  bool copy_row(uint16_t c, uint16_t y, int8_t *dst) {
    if (!valid || !dst || c >= tensor->C || y >= tensor->W) return false;
    if (!tensor->file_path) {
      memcpy(dst, tensor->buffer.data +
                  (static_cast<size_t>(c) * tensor->W + y) * tensor->W,
             tensor->W);
      return true;
    }
    int8_t first;
    if (!at(c, y, 0, &first)) return false;
    memcpy(dst, row, tensor->W);
    return true;
  }
#endif
};

// Buffered SD output: avoid one SPI filesystem call per activation byte.
// The 64-byte staging buffer is per invocation (no shared mutable globals).
struct QI8ActivationWriter {
  NoodleTensor *tensor;
  NDL_File file;
  int8_t *ram;
  uint8_t block[64];
  size_t fill;
  size_t written;
  size_t expected;
  bool valid;
  QI8ActivationWriter(NoodleTensor *t, uint16_t C, uint16_t W,
                      float scale, int32_t zp, bool vector = false)
      : tensor(t), ram(nullptr), fill(0), written(0),
        expected(static_cast<size_t>(C) * W * (vector ? 1u : W)), valid(false) {
    if (!t || !C || !W || !expected) return;
    if (t->file_path) {
      // Invalidate previous metadata before overwriting a file.
      t->rank = NOODLE_TENSOR_EMPTY;
      t->C = t->W = 0;
      // NOODLE_CASE2_SD_IO_EXPERIMENT_PATCH_V1: SdFat uses O_TRUNC; removing first is redundant.
#if !defined(NOODLE_USE_SDFAT) || !defined(NOODLE_INT8_SD_NO_PREDELETE) || !NOODLE_INT8_SD_NO_PREDELETE
      (void)noodle_fs_remove(t->file_path);
#endif
      file = noodle_fs_open_write(t->file_path);
      valid = static_cast<bool>(file);
    } else {
      ram = vector ? noodle_i8_tensor_require_1d(t, C, W, scale, zp)
                   : noodle_i8_tensor_require_2d(t, C, W, scale, zp);
      valid = ram != nullptr;
    }
  }
  ~QI8ActivationWriter() { if (file) file.close(); }
  bool flush() {
    if (!valid) return false;
    if (!file || !fill) return true;
    if (noodle_write_raw(file, block, fill) != fill) {
      valid = false;
      return false;
    }
    fill = 0;
    return true;
  }
  bool put(int8_t v) {
    if (!valid || written >= expected) return false;
    if (ram) ram[written++] = v;
    else {
      block[fill++] = static_cast<uint8_t>(v);
      ++written;
      if (fill == sizeof(block) && !flush()) return false;
    }
    return true;
  }
  bool complete(uint16_t C, uint16_t W, float scale, int32_t zp,
                bool vector = false) {
    if (!valid || written != expected || !flush()) return false;
    if (!tensor->file_path) return true;
    file.flush();
    file.close();
    return vector ? noodle_i8_tensor_file_1d(tensor, C, W, scale, zp)
                  : noodle_i8_tensor_file_2d(tensor, C, W, scale, zp);
  }
};

struct QI8FusedWeights {
  const int8_t *weight = nullptr;
  const int32_t *bias = nullptr;
  const int32_t *mult = nullptr;
  const int32_t *shift = nullptr;
  bool progmem = false;
  NDL_File wf, bf, mf, sf;
  bool has_bias_file = false;
  ~QI8FusedWeights() {
    if (wf) wf.close(); if (bf) bf.close();
    if (mf) mf.close(); if (sf) sf.close();
  }
  bool load(uint16_t oc, size_t filter_size, int8_t *dst, int32_t *b,
            int32_t *m, int32_t *sh) {
    if (wf) {
      if (q_read_i8_block(wf, dst, filter_size) != filter_size) return false;
      *b = has_bias_file ? noodle_read_i32(bf) : 0;
      *m = noodle_read_i32(mf);
      *sh = noodle_read_i32(sf);
    } else if (progmem) {
      const size_t base = static_cast<size_t>(oc) * filter_size;
      for (size_t k = 0; k < filter_size; ++k)
        dst[k] = noodle_pgm_i8(weight, static_cast<uint32_t>(base + k));
      *b = bias ? noodle_pgm_i32(bias, oc) : 0;
      *m = noodle_pgm_i32(mult, oc);
      *sh = noodle_pgm_i32(shift, oc);
    } else {
      memcpy(dst, weight + static_cast<size_t>(oc) * filter_size, filter_size);
      *b = bias ? bias[oc] : 0;
      *m = mult[oc];
      *sh = shift[oc];
    }
    return true;
  }
};

static bool q_fused_weights_init(const ConvMem &l, QI8FusedWeights *w) {
  if (!l.weight || !l.multiplier || !l.shift) return false;
  w->weight = l.weight; w->bias = l.bias;
  w->mult = l.multiplier; w->shift = l.shift;
  return true;
}
static bool q_fused_weights_init(const ConvProgmem &l, QI8FusedWeights *w) {
  if (!l.weight || !l.multiplier || !l.shift) return false;
  w->weight = l.weight; w->bias = l.bias;
  w->mult = l.multiplier; w->shift = l.shift; w->progmem = true;
  return true;
}
static bool q_fused_weights_init(const Conv &l, QI8FusedWeights *w) {
  if (!q_open_required(l.weight_fn, &w->wf) ||
      !q_open_required(l.multiplier_fn, &w->mf) ||
      !q_open_required(l.shift_fn, &w->sf)) return false;
  if (l.bias_fn) {
    w->has_bias_file = q_open_required(l.bias_fn, &w->bf);
    if (!w->has_bias_file) return false;
  }
  return true;
}

template <typename Accumulator>
static bool q_fused_conv_at_acc(QI8ActivationReader *r, const int8_t *filter,
                                uint16_t K, uint16_t P0, uint16_t S,
                                uint16_t oy, uint16_t ox, int32_t input_zp,
                                int32_t bias, int32_t *result) {
  Accumulator dot = 0;
  const uint16_t W = r->tensor->W;
  const uint32_t kernel_plane = static_cast<uint32_t>(K) * K;
  for (uint16_t ic = 0; ic < r->tensor->C; ++ic) {
    const int8_t *w = filter + static_cast<size_t>(ic) * kernel_plane;
    for (uint16_t ky = 0; ky < K; ++ky) {
      const int32_t iy = static_cast<int32_t>(oy) * S + ky - P0;
      if (iy < 0 || iy >= W) continue;
      for (uint16_t kx = 0; kx < K; ++kx) {
        const int32_t ix = static_cast<int32_t>(ox) * S + kx - P0;
        if (ix < 0 || ix >= W) continue;
        int8_t value;
        if (!r->at(ic, static_cast<uint16_t>(iy),
                   static_cast<uint16_t>(ix), &value)) return false;
        dot += static_cast<Accumulator>(static_cast<int32_t>(value) - input_zp) *
               static_cast<Accumulator>(w[static_cast<uint32_t>(ky) * K + kx]);
      }
    }
  }
  // Preserve bias semantics, including saturation, with a single INT64 add.
  *result = q_sat_i32(static_cast<int64_t>(dot) + bias);
  return true;
}

static bool q_fused_conv_at(QI8ActivationReader *r, const int8_t *filter,
                            uint16_t K, uint16_t P0, uint16_t S,
                            uint16_t oy, uint16_t ox, int32_t input_zp,
                            int32_t bias, int32_t *result, bool acc32_safe) {
#if !defined(__AVR__) || NOODLE_INT8_CONV_ACC32
  if (acc32_safe)
    return q_fused_conv_at_acc<int32_t>(r, filter, K, P0, S, oy, ox,
                                        input_zp, bias, result);
#else
  (void)acc32_safe;
#endif
  return q_fused_conv_at_acc<int64_t>(r, filter, K, P0, S, oy, ox,
                                      input_zp, bias, result);
}

#if NOODLE_INT8_SD_PLANE_FAST
// Same ordered MACs as q_fused_conv_at_acc(), but directly index one
// contiguous CHW plane-set in SRAM instead of calling reader.at() for
// every multiply. Accumulation and bias saturation are bit-for-bit equal.
template <typename Accumulator>
static int32_t q_fused_conv_plane_acc(const int8_t *input, uint16_t C,
                                      uint16_t W, const int8_t *filter,
                                      uint16_t K, uint16_t P0, uint16_t S,
                                      uint16_t oy, uint16_t ox,
                                      int32_t input_zp, int32_t bias) {
  Accumulator dot = 0;
  const size_t channel_plane = static_cast<size_t>(W) * W;
  const size_t kernel_plane = static_cast<size_t>(K) * K;
  for (uint16_t ic = 0; ic < C; ++ic) {
    const int8_t *src = input + static_cast<size_t>(ic) * channel_plane;
    const int8_t *wt = filter + static_cast<size_t>(ic) * kernel_plane;
    for (uint16_t ky = 0; ky < K; ++ky) {
      const int32_t iy = static_cast<int32_t>(oy) * S + ky - P0;
      if (iy < 0 || iy >= W) continue;
      for (uint16_t kx = 0; kx < K; ++kx) {
        const int32_t ix = static_cast<int32_t>(ox) * S + kx - P0;
        if (ix < 0 || ix >= W) continue;
        const int32_t value = src[static_cast<size_t>(iy) * W + ix];
        dot += static_cast<Accumulator>(value - input_zp) *
               static_cast<Accumulator>(wt[static_cast<size_t>(ky) * K + kx]);
      }
    }
  }
  return q_sat_i32(static_cast<int64_t>(dot) + bias);
}
#endif
static bool q_fused_conv_select(QI8ActivationReader *r,
                                const int8_t *plane, const int8_t *filter,
                                uint16_t K, uint16_t P0, uint16_t S,
                                uint16_t oy, uint16_t ox, int32_t input_zp,
                                int32_t bias, int32_t *result, bool acc32_safe) {
#if NOODLE_INT8_SD_PLANE_FAST
  if (plane) {
#if !defined(__AVR__) || NOODLE_INT8_CONV_ACC32
    if (acc32_safe) {
      *result = q_fused_conv_plane_acc<int32_t>(plane, r->tensor->C,
           r->tensor->W, filter, K, P0, S, oy, ox, input_zp, bias);
      return true;
    }
#endif
    *result = q_fused_conv_plane_acc<int64_t>(plane, r->tensor->C,
         r->tensor->W, filter, K, P0, S, oy, ox, input_zp, bias);
    return true;
  }
#else
  (void)plane;
#endif
  return q_fused_conv_at(r, filter, K, P0, S, oy, ox,
                         input_zp, bias, result, acc32_safe);
}

#if NOODLE_INT8_SD_POINTWISE_ROWBUF
template <typename Accumulator>
static int32_t q_fused_pw_row_dot(const int8_t *rows,
                                   const int8_t *filter, uint16_t channels,
                                   uint16_t W, uint16_t px,
                                   int32_t input_zp, int32_t bias) {
  Accumulator dot = 0;
  for (uint16_t ic = 0; ic < channels; ++ic) {
    const int32_t value = rows[static_cast<size_t>(ic) * W + px];
    dot += static_cast<Accumulator>(value - input_zp) *
           static_cast<Accumulator>(filter[ic]);
  }
  // Match q_fused_conv_at_acc(): saturate only after INT64 bias addition.
  return q_sat_i32(static_cast<int64_t>(dot) + bias);
}
#endif

template <typename Layer>
static uint16_t q_fused_conv2d(NoodleTensor *input, const Layer &l,
                               const NoodleI8Pool &pool, NoodleTensor *output) {
  if (!input || !output || input == output || input->rank != NOODLE_TENSOR_2D ||
      (!input->buffer.data && !input->file_path) || !l.O ||
      !q_input_matches(input, l.input_scale, l.input_zero_point) ||
      !pool.M || !pool.T) return 0;
  if (input->file_path && output->file_path &&
      strcmp(input->file_path, output->file_path) == 0) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, l.K, l.P, l.S, &g) || g.V < pool.M) return 0;
  const uint16_t outW = static_cast<uint16_t>((g.V - pool.M) / pool.T + 1);
  const uint64_t flen = static_cast<uint64_t>(input->C) * l.K * l.K;
  if (!flen || flen > static_cast<size_t>(-1)) return 0;
  const size_t filter_size = static_cast<size_t>(flen);
  const bool use_acc32 = q_fused_acc32_safe(flen, l.input_zero_point);
  QI8FusedWeights weights;
  if (!q_fused_weights_init(l, &weights)) return 0;
  int8_t *filter = static_cast<int8_t *>(q_alloc(filter_size));
  if (!filter) return 0;
  struct FilterGuard { int8_t *p; ~FilterGuard(){ free(p); } } filter_guard{filter};
  QI8ActivationReader reader(input);
  if (!reader.valid) return 0;
  const int8_t *prefetched = nullptr;
#if NOODLE_INT8_SD_PLANE_FAST
  if (l.K > 1 && reader.enable_plane(false))
    prefetched = reader.plane_data;  // entire CHW input, when within budget
  if (!reader.valid) return 0;
#endif
#if NOODLE_INT8_SD_KROW_CACHE
  // Cache rows only if the complete input could not be prefetched.
  if (l.K > 1 && !prefetched) (void)reader.enable_krow(l.K, false);
#endif
  QI8ActivationWriter writer(output, l.O, outW, l.output_scale,
                            l.output_zero_point);
  if (!writer.valid) return 0;

#if NOODLE_INT8_SD_POINTWISE_ROWBUF
  // Optimized only for file-backed 1x1 convolution with identity pooling.
  // Read C whole rows before computing W output pixels. Preserve CHW order.
  const bool is_sd_pw = input->file_path && l.K == 1 && l.S == 1 &&
                        g.P0 == 0 && g.V == input->W &&
                        pool.M == 1 && pool.T == 1 && outW == input->W;
  const uint32_t row_bytes32 = static_cast<uint32_t>(input->C) * input->W;
  int8_t *rows = (is_sd_pw && row_bytes32 &&
                  row_bytes32 <= static_cast<size_t>(-1))
      ? static_cast<int8_t *>(q_alloc(static_cast<size_t>(row_bytes32)))
      : nullptr;
  // Low-memory fallback: use existing one-row kernel if allocation fails.
  struct RowGuard { int8_t *p; ~RowGuard() { free(p); } } row_guard{rows};
#endif

  for (uint16_t oc = 0; oc < l.O; ++oc) {
    int32_t bias, mult, shift;
    if (!weights.load(oc, filter_size, filter, &bias, &mult, &shift)) return 0;
#if NOODLE_INT8_SD_POINTWISE_ROWBUF
    if (rows) {
      for (uint16_t py = 0; py < outW; ++py) {
        for (uint16_t ic = 0; ic < input->C; ++ic) {
          if (!reader.copy_row(ic, py, rows + static_cast<size_t>(ic) * input->W))
            return 0;
        }
        for (uint16_t px = 0; px < outW; ++px) {
          int32_t acc;
#if !defined(__AVR__) || NOODLE_INT8_CONV_ACC32
          if (use_acc32)
            acc = q_fused_pw_row_dot<int32_t>(rows, filter, input->C, input->W,
                                               px, l.input_zero_point, bias);
          else
#endif
            acc = q_fused_pw_row_dot<int64_t>(rows, filter, input->C, input->W,
                                               px, l.input_zero_point, bias);
          const int8_t q = static_cast<int8_t>(q_apply_output(
              acc, mult, shift, l.output_zero_point,
              l.activation_min, l.activation_max));
          // With 1x1 identity pooling MAX and MEAN both return q exactly.
          if (!writer.put(q)) return 0;
        }
      }
      continue;
    }
#endif
    for (uint16_t py = 0; py < outW; ++py) {
      for (uint16_t px = 0; px < outW; ++px) {
#if defined(NOODLE_INT8_FUSED_IDENTITY_FAST) && NOODLE_INT8_FUSED_IDENTITY_FAST
        // NOODLE_CASE2_SD_IO_EXPERIMENT_PATCH_V1: fused identity pool is just quantized convolution.
        if (pool.M == 1 && pool.T == 1) {
          int32_t acc;
          if (!q_fused_conv_select(&reader, prefetched, filter, l.K, g.P0, l.S, py, px,
                               l.input_zero_point, bias, &acc, use_acc32)) return 0;
          const int8_t q = static_cast<int8_t>(q_apply_output(
              acc, mult, shift, l.output_zero_point,
              l.activation_min, l.activation_max));
          if (!writer.put(q)) return 0;
          continue;
        }
#endif
        int8_t vmax = -128;
        int32_t sum = 0;
        for (uint16_t ky = 0; ky < pool.M; ++ky) {
          for (uint16_t kx = 0; kx < pool.M; ++kx) {
            int32_t acc;
            if (!q_fused_conv_select(&reader, prefetched, filter, l.K, g.P0, l.S,
                                 static_cast<uint16_t>(py * pool.T + ky),
                                 static_cast<uint16_t>(px * pool.T + kx),
                                 l.input_zero_point, bias, &acc,
                                 use_acc32)) return 0;
            const int8_t q = static_cast<int8_t>(q_apply_output(
                acc, mult, shift, l.output_zero_point,
                l.activation_min, l.activation_max));
            if (q > vmax) vmax = q;
            sum += static_cast<int32_t>(q) - l.output_zero_point;
          }
        }
        int8_t pooled = vmax;
        if (pool.mode == NOODLE_I8_POOL_MEAN) {
          const int32_t n = static_cast<int32_t>(pool.M) * pool.M;
          const int32_t rounded = sum >= 0 ? (sum + n / 2) / n
                                           : (sum - n / 2) / n;
          pooled = noodle_i8_clamp(rounded + l.output_zero_point);
        }
        if (!writer.put(pooled)) return 0;
      }
    }
  }
  if (!writer.complete(l.O, outW, l.output_scale,
                       l.output_zero_point)) return 0;
  return outW;
}

uint16_t noodle_i8_conv2d_fused(NoodleTensor *in, const ConvMem &l,
                                 const NoodleI8Pool &p, NoodleTensor *out) {
  return q_fused_conv2d(in, l, p, out);
}
uint16_t noodle_i8_conv2d_fused(NoodleTensor *in, const Conv &l,
                                 const NoodleI8Pool &p, NoodleTensor *out) {
  return q_fused_conv2d(in, l, p, out);
}
uint16_t noodle_i8_conv2d_fused(NoodleTensor *in, const ConvProgmem &l,
                                 const NoodleI8Pool &p, NoodleTensor *out) {
  return q_fused_conv2d(in, l, p, out);
}



// Fused depthwise convolution-pooling uses the same file reader and streaming
// writer as ordinary Conv2D. Each output channel depends on just one input
// channel, even when depth_multiplier > 1.
template <typename Accumulator>
static bool q_fused_dw_at_acc(QI8ActivationReader *r, uint16_t ic,
                              const int8_t *filter, uint16_t K,
                              uint16_t P0, uint16_t S, uint16_t oy, uint16_t ox,
                              int32_t input_zp, int32_t bias, int32_t *result) {
  Accumulator dot = 0;
  const uint16_t W = r->tensor->W;
  for (uint16_t ky = 0; ky < K; ++ky) {
    const int32_t iy = static_cast<int32_t>(oy) * S + ky - P0;
    if (iy < 0 || iy >= W) continue;
    for (uint16_t kx = 0; kx < K; ++kx) {
      const int32_t ix = static_cast<int32_t>(ox) * S + kx - P0;
      if (ix < 0 || ix >= W) continue;
      int8_t value;
      if (!r->at(ic, static_cast<uint16_t>(iy),
                 static_cast<uint16_t>(ix), &value)) return false;
      dot += static_cast<Accumulator>(static_cast<int32_t>(value) - input_zp) *
             static_cast<Accumulator>(filter[static_cast<uint32_t>(ky) * K + kx]);
    }
  }
  *result = q_sat_i32(static_cast<int64_t>(dot) + bias);
  return true;
}

static bool q_fused_dw_at(QI8ActivationReader *r, uint16_t ic,
                          const int8_t *filter, uint16_t K,
                          uint16_t P0, uint16_t S, uint16_t oy, uint16_t ox,
                          int32_t input_zp, int32_t bias, int32_t *result,
                          bool acc32_safe) {
#if !defined(__AVR__) || NOODLE_INT8_CONV_ACC32
  if (acc32_safe)
    return q_fused_dw_at_acc<int32_t>(r, ic, filter, K, P0, S, oy, ox,
                                      input_zp, bias, result);
#else
  (void)acc32_safe;
#endif
  return q_fused_dw_at_acc<int64_t>(r, ic, filter, K, P0, S, oy, ox,
                                    input_zp, bias, result);
}

#if NOODLE_INT8_SD_PLANE_FAST
// The depthwise fast path reads one channel plane and performs direct
// indexing. Support arbitrary K, stride, padding and depth multiplier.
template <typename Accumulator>
static int32_t q_fused_dw_plane_acc(const int8_t *src, uint16_t W,
                                    const int8_t *filter, uint16_t K,
                                    uint16_t P0, uint16_t S, uint16_t oy,
                                    uint16_t ox, int32_t input_zp,
                                    int32_t bias) {
  Accumulator dot = 0;
  for (uint16_t ky = 0; ky < K; ++ky) {
    const int32_t iy = static_cast<int32_t>(oy) * S + ky - P0;
    if (iy < 0 || iy >= W) continue;
    for (uint16_t kx = 0; kx < K; ++kx) {
      const int32_t ix = static_cast<int32_t>(ox) * S + kx - P0;
      if (ix < 0 || ix >= W) continue;
      const int32_t value = src[static_cast<size_t>(iy) * W + ix];
      dot += static_cast<Accumulator>(value - input_zp) *
             static_cast<Accumulator>(filter[static_cast<size_t>(ky) * K + kx]);
    }
  }
  return q_sat_i32(static_cast<int64_t>(dot) + bias);
}
#endif
static bool q_fused_dw_select(QI8ActivationReader *r, const int8_t *plane,
                              uint16_t ic, const int8_t *filter, uint16_t K,
                              uint16_t P0, uint16_t S, uint16_t oy,
                              uint16_t ox, int32_t input_zp,
                              int32_t bias, int32_t *result, bool acc32_safe) {
#if NOODLE_INT8_SD_PLANE_FAST
  if (plane) {
#if !defined(__AVR__) || NOODLE_INT8_CONV_ACC32
    if (acc32_safe) {
      *result = q_fused_dw_plane_acc<int32_t>(plane, r->tensor->W,
                          filter, K, P0, S, oy, ox, input_zp, bias);
      return true;
    }
#endif
    *result = q_fused_dw_plane_acc<int64_t>(plane, r->tensor->W,
                        filter, K, P0, S, oy, ox, input_zp, bias);
    return true;
  }
#else
  (void)plane;
#endif
  return q_fused_dw_at(r, ic, filter, K, P0, S, oy, ox,
                       input_zp, bias, result, acc32_safe);
}

template <typename Layer>
static uint16_t q_fused_dwconv2d(NoodleTensor *input, const Layer &l,
                                 const NoodleI8Pool &pool, NoodleTensor *output) {
  if (!input || !output || input == output || input->rank != NOODLE_TENSOR_2D ||
      (!input->buffer.data && !input->file_path) || !l.depth_multiplier ||
      !q_input_matches(input, l.input_scale, l.input_zero_point) ||
      !pool.M || !pool.T) return 0;
  if (input->file_path && output->file_path &&
      strcmp(input->file_path, output->file_path) == 0) return 0;
  const uint32_t total = static_cast<uint32_t>(input->C) * l.depth_multiplier;
  if (total > UINT16_MAX || (l.O && l.O != total)) return 0;
  const uint16_t O = static_cast<uint16_t>(total);
  QGeom g;
  if (!q_conv_geom(input->W, l.K, l.P, l.S, &g) || g.V < pool.M) return 0;
  const uint16_t outW = static_cast<uint16_t>((g.V - pool.M) / pool.T + 1);
  const uint64_t klen = static_cast<uint64_t>(l.K) * l.K;
  if (klen > static_cast<size_t>(-1)) return 0;
  const bool use_acc32 = q_fused_acc32_safe(klen, l.input_zero_point);
  QI8FusedWeights weights;
  if (!q_fused_weights_init(l, &weights)) return 0;
  int8_t *filter = static_cast<int8_t *>(q_alloc(static_cast<size_t>(klen)));
  if (!filter) return 0;
  struct Guard { int8_t *p; ~Guard(){ free(p); } } guard{filter};
  QI8ActivationReader reader(input);
  if (!reader.valid) return 0;
#if NOODLE_INT8_SD_PLANE_FAST
  const bool use_plane = l.K > 1 && reader.enable_plane(true);
  if (!reader.valid) return 0;
#else
  const bool use_plane = false;
#endif
#if NOODLE_INT8_SD_KROW_CACHE
  // Depthwise only needs K rows from the current channel.
  if (l.K > 1 && !use_plane) (void)reader.enable_krow(l.K, true);
#endif
  QI8ActivationWriter writer(output, O, outW, l.output_scale,
                            l.output_zero_point);
  if (!writer.valid) return 0;
  for (uint16_t oc = 0; oc < O; ++oc) {
    int32_t bias, mult, shift;
    if (!weights.load(oc, static_cast<size_t>(klen), filter,
                      &bias, &mult, &shift)) return 0;
    const uint16_t ic = static_cast<uint16_t>(oc / l.depth_multiplier);
    const int8_t *prefetched = nullptr;
#if NOODLE_INT8_SD_PLANE_FAST
    if (use_plane) {
      prefetched = reader.channel_plane(ic);
      if (!prefetched) return 0;
    }
#endif
    for (uint16_t py = 0; py < outW; ++py)
      for (uint16_t px = 0; px < outW; ++px) {
#if defined(NOODLE_INT8_FUSED_IDENTITY_FAST) && NOODLE_INT8_FUSED_IDENTITY_FAST
        // NOODLE_CASE2_SD_IO_EXPERIMENT_PATCH_V1: identity pooling, preserving quantization exactly.
        if (pool.M == 1 && pool.T == 1) {
          int32_t acc;
          if (!q_fused_dw_select(&reader, prefetched, ic, filter, l.K, g.P0, l.S, py, px,
                             l.input_zero_point, bias, &acc, use_acc32)) return 0;
          const int8_t q = static_cast<int8_t>(q_apply_output(
              acc, mult, shift, l.output_zero_point,
              l.activation_min, l.activation_max));
          if (!writer.put(q)) return 0;
          continue;
        }
#endif
        int8_t vmax = -128;
        int32_t sum = 0;
        for (uint16_t ky = 0; ky < pool.M; ++ky)
          for (uint16_t kx = 0; kx < pool.M; ++kx) {
            int32_t acc;
            if (!q_fused_dw_select(&reader, prefetched, ic, filter, l.K, g.P0, l.S,
                                static_cast<uint16_t>(py * pool.T + ky),
                                static_cast<uint16_t>(px * pool.T + kx),
                                l.input_zero_point, bias, &acc,
                                use_acc32)) return 0;
            const int8_t q = static_cast<int8_t>(q_apply_output(
                acc, mult, shift, l.output_zero_point,
                l.activation_min, l.activation_max));
            if (q > vmax) vmax = q;
            sum += static_cast<int32_t>(q) - l.output_zero_point;
          }
        int8_t pooled = vmax;
        if (pool.mode == NOODLE_I8_POOL_MEAN) {
          const int32_t n = static_cast<int32_t>(pool.M) * pool.M;
          const int32_t rounded = sum >= 0 ? (sum + n / 2) / n
                                           : (sum - n / 2) / n;
          pooled = noodle_i8_clamp(rounded + l.output_zero_point);
        }
        if (!writer.put(pooled)) return 0;
      }
  }
  if (!writer.complete(O, outW, l.output_scale,
                       l.output_zero_point)) return 0;
  return outW;
}

uint16_t noodle_i8_dwconv2d_fused(NoodleTensor *i, const ConvMem &l,
                                   const NoodleI8Pool &p, NoodleTensor *o) {
  return q_fused_dwconv2d(i, l, p, o);
}
uint16_t noodle_i8_dwconv2d_fused(NoodleTensor *i, const Conv &l,
                                   const NoodleI8Pool &p, NoodleTensor *o) {
  return q_fused_dwconv2d(i, l, p, o);
}
uint16_t noodle_i8_dwconv2d_fused(NoodleTensor *i, const ConvProgmem &l,
                                   const NoodleI8Pool &p, NoodleTensor *o) {
  return q_fused_dwconv2d(i, l, p, o);
}

// Separate pooling on file-backed tensors; no full activation materialization.
uint16_t noodle_i8_pool2d_file(NoodleTensor *input, const NoodleI8Pool &pool,
                                NoodleTensor *output) {
  if (!input || !output || input == output ||
      input->rank != NOODLE_TENSOR_2D ||
      (!input->file_path && !input->buffer.data) || !pool.M || !pool.T ||
      input->W < pool.M ||
      (input->file_path && output->file_path &&
       strcmp(input->file_path, output->file_path) == 0)) return 0;
  const uint16_t W = static_cast<uint16_t>((input->W - pool.M) / pool.T + 1);
  QI8ActivationReader reader(input);
  if (!reader.valid) return 0;
  QI8ActivationWriter writer(output, input->C, W, input->scale, input->zero_point);
  if (!writer.valid) return 0;
  for (uint16_t c = 0; c < input->C; ++c)
    for (uint16_t y = 0; y < W; ++y)
      for (uint16_t x = 0; x < W; ++x) {
        int8_t vmax = -128;
        int32_t sum = 0;
        for (uint16_t ky = 0; ky < pool.M; ++ky)
          for (uint16_t kx = 0; kx < pool.M; ++kx) {
            int8_t q;
            if (!reader.at(c, static_cast<uint16_t>(y * pool.T + ky),
                           static_cast<uint16_t>(x * pool.T + kx), &q)) return 0;
            if (q > vmax) vmax = q;
            sum += static_cast<int32_t>(q) - input->zero_point;
          }
        int8_t value = vmax;
        if (pool.mode == NOODLE_I8_POOL_MEAN) {
          const int32_t n = static_cast<int32_t>(pool.M) * pool.M;
          const int32_t rounded = sum >= 0 ? (sum + n / 2) / n
                                           : (sum - n / 2) / n;
          value = noodle_i8_clamp(rounded + input->zero_point);
        }
        if (!writer.put(value)) return 0;
      }
  if (!writer.complete(input->C, W, input->scale,
                       input->zero_point)) return 0;
  return W;
}

// Preserve the original CHW -> pixel-major flattened tensor layout without
// allocating the 1D intermediate when the target is file-backed.
uint16_t noodle_i8_flat_file(NoodleTensor *input, NoodleTensor *output) {
  if (!input || !output || input == output ||
      input->rank != NOODLE_TENSOR_2D ||
      (!input->file_path && !input->buffer.data) ||
      (input->file_path && output->file_path &&
       strcmp(input->file_path, output->file_path) == 0)) return 0;
  const size_t count = static_cast<size_t>(input->C) * input->W * input->W;
  if (!count || count > UINT16_MAX) return 0;
  QI8ActivationReader reader(input);
  if (!reader.valid) return 0;
  QI8ActivationWriter writer(output, static_cast<uint16_t>(count), 1,
                             input->scale, input->zero_point, true);
  if (!writer.valid) return 0;
  for (uint16_t y = 0; y < input->W; ++y)
    for (uint16_t x = 0; x < input->W; ++x)
      for (uint16_t c = 0; c < input->C; ++c) {
        int8_t v;
        if (!reader.at(c, y, x, &v) || !writer.put(v)) return 0;
      }
  if (!writer.complete(static_cast<uint16_t>(count), 1,
                       input->scale, input->zero_point, true)) return 0;
  return static_cast<uint16_t>(count);
}

uint16_t noodle_i8_conv2d(NoodleTensor *input, const ConvMem &layer,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  const uint32_t filter_size = static_cast<uint32_t>(input->C) * layer.K * layer.K;
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
#if NOODLE_INT8_PW_IMPL
  if (layer.K == 1) {
    int32_t bias_block[NOODLE_INT8_PW_OUTPUT_BLOCK];
    int32_t multiplier_block[NOODLE_INT8_PW_OUTPUT_BLOCK];
    int32_t shift_block[NOODLE_INT8_PW_OUTPUT_BLOCK];
    const uint16_t block_limit = static_cast<uint16_t>(NOODLE_INT8_PW_OUTPUT_BLOCK);
    for (uint16_t oc0 = 0; oc0 < layer.O; oc0 = static_cast<uint16_t>(oc0 + block_limit)) {
      const uint16_t block_o = (oc0 + block_limit < layer.O)
                                 ? block_limit
                                 : static_cast<uint16_t>(layer.O - oc0);
      const int8_t *filter_block = layer.weight +
          static_cast<size_t>(oc0) * input->C;
      for (uint16_t bo = 0; bo < block_o; ++bo) {
        const uint16_t oc = static_cast<uint16_t>(oc0 + bo);
        bias_block[bo] = layer.bias ? layer.bias[oc] : 0;
        multiplier_block[bo] = layer.multiplier[oc];
        shift_block[bo] = layer.shift[oc];
      }
      if (!q_conv2d_pw_blocked(
              input, g.V, g.P0, layer.S, filter_block, block_o,
              bias_block, multiplier_block, shift_block,
              layer.input_zero_point, layer.output_zero_point,
              layer.activation_min, layer.activation_max,
              dst + static_cast<uint32_t>(oc0) * out_plane)) return 0;
    }
    return g.V;
  }
#endif
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    const int8_t *filter = layer.weight + static_cast<uint32_t>(oc) * filter_size;
    const int32_t bias = layer.bias ? layer.bias[oc] : 0;
    if (!q_conv2d_filter(input, g.V, g.P0, filter, bias,
                         layer.multiplier[oc], layer.shift[oc],
                         layer.K, layer.S, layer.input_zero_point,
                         layer.output_zero_point, layer.activation_min,
                         layer.activation_max,
                         dst + static_cast<uint32_t>(oc) * out_plane)) return 0;
  }
  return g.V;
}

uint16_t noodle_i8_conv2d(NoodleTensor *input, const Conv &layer,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;

  NDL_File wf, bf, mf, sf;
  if (!q_open_required(layer.weight_fn, &wf) ||
      !q_open_required(layer.multiplier_fn, &mf) ||
      !q_open_required(layer.shift_fn, &sf)) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const bool has_bias = layer.bias_fn && q_open_required(layer.bias_fn, &bf);
  if (layer.bias_fn && !has_bias) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }

#if NOODLE_INT8_PW_IMPL
  if (layer.K == 1) {
    const size_t block_filter_size = static_cast<size_t>(input->C) *
                                     NOODLE_INT8_PW_OUTPUT_BLOCK;
    int8_t *filter_block = q_filter_require(block_filter_size);
    if (!filter_block) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    int32_t bias_block[NOODLE_INT8_PW_OUTPUT_BLOCK];
    int32_t multiplier_block[NOODLE_INT8_PW_OUTPUT_BLOCK];
    int32_t shift_block[NOODLE_INT8_PW_OUTPUT_BLOCK];
    const uint16_t block_limit = static_cast<uint16_t>(NOODLE_INT8_PW_OUTPUT_BLOCK);
    const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
    for (uint16_t oc0 = 0; oc0 < layer.O; oc0 = static_cast<uint16_t>(oc0 + block_limit)) {
      const uint16_t block_o = (oc0 + block_limit < layer.O)
                                 ? block_limit
                                 : static_cast<uint16_t>(layer.O - oc0);
      const size_t bytes = static_cast<size_t>(block_o) * input->C;
      if (q_read_i8_block(wf, filter_block, bytes) != bytes) {
        q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
      }
      for (uint16_t bo = 0; bo < block_o; ++bo) {
        bias_block[bo] = has_bias ? noodle_read_i32(bf) : 0;
        multiplier_block[bo] = noodle_read_i32(mf);
        shift_block[bo] = noodle_read_i32(sf);
      }
      if (!q_conv2d_pw_blocked(
              input, g.V, g.P0, layer.S, filter_block, block_o,
              bias_block, multiplier_block, shift_block,
              layer.input_zero_point, layer.output_zero_point,
              layer.activation_min, layer.activation_max,
              dst + static_cast<uint32_t>(oc0) * out_plane)) {
        q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
      }
    }
    q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
    return g.V;
  }
#endif

  const size_t filter_size = static_cast<size_t>(input->C) * layer.K * layer.K;
  int8_t *filter = q_filter_require(filter_size);
  if (!filter) {
    q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    if (q_read_i8_block(wf, filter, filter_size) != filter_size) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    const int32_t bias = has_bias ? noodle_read_i32(bf) : 0;
    const int32_t mult = noodle_read_i32(mf);
    const int32_t shift = noodle_read_i32(sf);
    if (!q_conv2d_filter(input, g.V, g.P0, filter, bias, mult, shift,
                         layer.K, layer.S, layer.input_zero_point,
                         layer.output_zero_point, layer.activation_min,
                         layer.activation_max,
                         dst + static_cast<uint32_t>(oc) * out_plane)) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
  }
  q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
  return g.V;
}

static int32_t q_conv1d_at(const int8_t *input, uint16_t I, uint16_t W,
                           const int8_t *filter, uint16_t K, uint16_t o,
                           uint16_t P, uint16_t S, int32_t input_zp,
                           int32_t bias) {
  int64_t acc = bias;
  for (uint16_t ic = 0; ic < I; ++ic) {
    const int8_t *x = input + static_cast<size_t>(ic) * W;
    const int8_t *w = filter + static_cast<size_t>(ic) * K;
    for (uint16_t k = 0; k < K; ++k) {
      const int32_t ix = static_cast<int32_t>(o) * S + k - P;
      if (ix < 0 || ix >= W) continue;
      acc += static_cast<int64_t>(static_cast<int32_t>(x[ix]) - input_zp) * w[k];
    }
  }
  return q_sat_i32(acc);
}

uint16_t noodle_i8_conv1d(NoodleTensor *input, const ConvMem &layer,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_1D || !layer.weight ||
      !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_1d(output, layer.O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    const int8_t *filter = layer.weight + static_cast<size_t>(oc) * filter_size;
    const int32_t bias = layer.bias ? layer.bias[oc] : 0;
    for (uint16_t o = 0; o < g.V; ++o) {
      const int32_t acc = q_conv1d_at(input->buffer.data, input->C, input->W,
                                      filter, layer.K, o, g.P0, layer.S,
                                      layer.input_zero_point, bias);
      dst[static_cast<size_t>(oc) * g.V + o] = static_cast<int8_t>(
          q_apply_output(acc, layer.multiplier[oc], layer.shift[oc],
                         layer.output_zero_point, layer.activation_min,
                         layer.activation_max));
    }
  }
  return g.V;
}

uint16_t noodle_i8_conv1d(NoodleTensor *input, const Conv &layer,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_1D || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_1d(output, layer.O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  NDL_File wf, bf, mf, sf;
  if (!q_open_required(layer.weight_fn, &wf) ||
      !q_open_required(layer.multiplier_fn, &mf) ||
      !q_open_required(layer.shift_fn, &sf)) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const bool has_bias = layer.bias_fn && q_open_required(layer.bias_fn, &bf);
  if (layer.bias_fn && !has_bias) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K;
  int8_t *filter = q_filter_require(filter_size);
  if (!filter) { q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0; }
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    if (q_read_i8_block(wf, filter, filter_size) != filter_size) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    const int32_t bias = has_bias ? noodle_read_i32(bf) : 0;
    const int32_t mult = noodle_read_i32(mf);
    const int32_t shift = noodle_read_i32(sf);
    for (uint16_t o = 0; o < g.V; ++o) {
      const int32_t acc = q_conv1d_at(input->buffer.data, input->C, input->W,
                                      filter, layer.K, o, g.P0, layer.S,
                                      layer.input_zero_point, bias);
      dst[static_cast<size_t>(oc) * g.V + o] = static_cast<int8_t>(
          q_apply_output(acc, mult, shift, layer.output_zero_point,
                         layer.activation_min, layer.activation_max));
    }
  }
  q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
  return g.V;
}

static int32_t q_dw_at(const NoodleTensor *input, uint16_t ic,
                       const int8_t *kernel, uint16_t K,
                       uint16_t oy, uint16_t ox, uint16_t P0, uint16_t S,
                       int32_t input_zp, int32_t bias) {
  const uint32_t plane = static_cast<uint32_t>(input->W) * input->W;
  const int8_t *x = input->buffer.data + static_cast<uint32_t>(ic) * plane;
  int64_t acc = bias;
  for (uint16_t ky = 0; ky < K; ++ky) {
    const int32_t iy = static_cast<int32_t>(oy) * S + ky - P0;
    if (iy < 0 || iy >= input->W) continue;
    for (uint16_t kx = 0; kx < K; ++kx) {
      const int32_t ix = static_cast<int32_t>(ox) * S + kx - P0;
      if (ix < 0 || ix >= input->W) continue;
      const int32_t xv = static_cast<int32_t>(x[static_cast<uint32_t>(iy) * input->W + ix]) - input_zp;
      acc += static_cast<int64_t>(xv) * kernel[static_cast<uint32_t>(ky) * K + kx];
    }
  }
  return q_sat_i32(acc);
}

uint16_t noodle_i8_dwconv2d(NoodleTensor *input, const ConvMem &layer,
                          NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || layer.depth_multiplier == 0 ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint32_t expected_o = static_cast<uint32_t>(input->C) * layer.depth_multiplier;
  const uint16_t O = layer.O ? layer.O : static_cast<uint16_t>(expected_o);
  if (expected_o > 65535u || O != expected_o) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  const uint32_t ksize = static_cast<uint32_t>(layer.K) * layer.K;
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < O; ++oc) {
    const uint16_t ic = oc / layer.depth_multiplier;
    const int8_t *kernel = layer.weight + static_cast<uint32_t>(oc) * ksize;
    const int32_t bias = layer.bias ? layer.bias[oc] : 0;
    int8_t *out = dst + static_cast<uint32_t>(oc) * out_plane;
    for (uint16_t oy = 0; oy < g.V; ++oy)
      for (uint16_t ox = 0; ox < g.V; ++ox) {
        const int32_t acc = q_dw_at(input, ic, kernel, layer.K, oy, ox,
                                    g.P0, layer.S, layer.input_zero_point, bias);
        out[static_cast<uint32_t>(oy) * g.V + ox] = static_cast<int8_t>(
            q_apply_output(acc, layer.multiplier[oc], layer.shift[oc],
                           layer.output_zero_point, layer.activation_min,
                           layer.activation_max));
      }
  }
  return g.V;
}

uint16_t noodle_i8_dwconv2d(NoodleTensor *input, const Conv &layer,
                          NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || layer.depth_multiplier == 0 ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint32_t expected_o = static_cast<uint32_t>(input->C) * layer.depth_multiplier;
  const uint16_t O = layer.O ? layer.O : static_cast<uint16_t>(expected_o);
  if (expected_o > 65535u || O != expected_o) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  NDL_File wf, bf, mf, sf;
  if (!q_open_required(layer.weight_fn, &wf) ||
      !q_open_required(layer.multiplier_fn, &mf) ||
      !q_open_required(layer.shift_fn, &sf)) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const bool has_bias = layer.bias_fn && q_open_required(layer.bias_fn, &bf);
  if (layer.bias_fn && !has_bias) { q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0; }
  const size_t ksize = static_cast<size_t>(layer.K) * layer.K;
  int8_t *kernel = q_filter_require(ksize);
  if (!kernel) { q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0; }
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < O; ++oc) {
    if (q_read_i8_block(wf, kernel, ksize) != ksize) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    const int32_t bias = has_bias ? noodle_read_i32(bf) : 0;
    const int32_t mult = noodle_read_i32(mf);
    const int32_t shift = noodle_read_i32(sf);
    const uint16_t ic = oc / layer.depth_multiplier;
    int8_t *out = dst + static_cast<uint32_t>(oc) * out_plane;
    for (uint16_t oy = 0; oy < g.V; ++oy)
      for (uint16_t ox = 0; ox < g.V; ++ox) {
        const int32_t acc = q_dw_at(input, ic, kernel, layer.K, oy, ox,
                                    g.P0, layer.S, layer.input_zero_point, bias);
        out[static_cast<uint32_t>(oy) * g.V + ox] = static_cast<int8_t>(
            q_apply_output(acc, mult, shift, layer.output_zero_point,
                           layer.activation_min, layer.activation_max));
      }
  }
  q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
  return g.V;
}

static uint16_t q_transpose_width(uint16_t W, uint16_t K, uint16_t P,
                                  uint16_t S, uint16_t OP) {
  if (!W || !K || !S || P == NOODLE_SAME_PADDING) return 0;
  const int64_t v = (static_cast<int64_t>(W) - 1) * S - 2 * P + K + OP;
  return v > 0 && v <= 65535 ? static_cast<uint16_t>(v) : 0;
}

static bool q_transpose_filter(const NoodleTensor *input, uint16_t V,
                               const int8_t *filter, uint16_t K,
                               uint16_t P, uint16_t S, int32_t input_zp,
                               int32_t bias, int32_t mult, int32_t shift,
                               int32_t output_zp, int32_t act_min,
                               int32_t act_max, int8_t *out) {
  const uint32_t plane = static_cast<uint32_t>(V) * V;
  int32_t *acc = q_acc_require(plane);
  if (!acc) return false;
  for (uint32_t i = 0; i < plane; ++i) acc[i] = bias;
  const uint32_t in_plane = static_cast<uint32_t>(input->W) * input->W;
  const uint32_t kplane = static_cast<uint32_t>(K) * K;
  for (uint16_t ic = 0; ic < input->C; ++ic) {
    const int8_t *x = input->buffer.data + static_cast<uint32_t>(ic) * in_plane;
    const int8_t *w = filter + static_cast<uint32_t>(ic) * kplane;
    for (uint16_t iy = 0; iy < input->W; ++iy)
      for (uint16_t ix = 0; ix < input->W; ++ix) {
        const int32_t xv = static_cast<int32_t>(x[static_cast<uint32_t>(iy) * input->W + ix]) - input_zp;
        for (uint16_t ky = 0; ky < K; ++ky) {
          const int32_t oy = static_cast<int32_t>(iy) * S + ky - P;
          if (oy < 0 || oy >= V) continue;
          for (uint16_t kx = 0; kx < K; ++kx) {
            const int32_t ox = static_cast<int32_t>(ix) * S + kx - P;
            if (ox < 0 || ox >= V) continue;
            const uint32_t oi = static_cast<uint32_t>(oy) * V + ox;
            const int64_t next = static_cast<int64_t>(acc[oi]) +
                static_cast<int64_t>(xv) * w[static_cast<uint32_t>(ky) * K + kx];
            acc[oi] = q_sat_i32(next);
          }
        }
      }
  }
  for (uint32_t i = 0; i < plane; ++i)
    out[i] = static_cast<int8_t>(q_apply_output(acc[i], mult, shift,
                                                output_zp, act_min, act_max));
  return true;
}

uint16_t noodle_i8_conv_transpose2d(NoodleTensor *input,
                                  const ConvMem &layer,
                                  NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint16_t V = q_transpose_width(input->W, layer.K, layer.P, layer.S, layer.OP);
  if (!V) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K * layer.K;
  const uint32_t plane = static_cast<uint32_t>(V) * V;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    if (!q_transpose_filter(input, V, layer.weight + static_cast<size_t>(oc) * filter_size,
                            layer.K, layer.P, layer.S, layer.input_zero_point,
                            layer.bias ? layer.bias[oc] : 0,
                            layer.multiplier[oc], layer.shift[oc],
                            layer.output_zero_point, layer.activation_min,
                            layer.activation_max,
                            dst + static_cast<uint32_t>(oc) * plane)) return 0;
  }
  return V;
}

uint16_t noodle_i8_conv_transpose2d(NoodleTensor *input,
                                  const Conv &layer,
                                  NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint16_t V = q_transpose_width(input->W, layer.K, layer.P, layer.S, layer.OP);
  if (!V) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  NDL_File wf, bf, mf, sf;
  if (!q_open_required(layer.weight_fn, &wf) ||
      !q_open_required(layer.multiplier_fn, &mf) ||
      !q_open_required(layer.shift_fn, &sf)) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const bool has_bias = layer.bias_fn && q_open_required(layer.bias_fn, &bf);
  if (layer.bias_fn && !has_bias) { q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0; }
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K * layer.K;
  int8_t *filter = q_filter_require(filter_size);
  if (!filter) { q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0; }
  const uint32_t plane = static_cast<uint32_t>(V) * V;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    if (q_read_i8_block(wf, filter, filter_size) != filter_size) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    const int32_t bias = has_bias ? noodle_read_i32(bf) : 0;
    const int32_t mult = noodle_read_i32(mf);
    const int32_t shift = noodle_read_i32(sf);
    if (!q_transpose_filter(input, V, filter, layer.K, layer.P, layer.S,
                            layer.input_zero_point, bias, mult, shift,
                            layer.output_zero_point, layer.activation_min,
                            layer.activation_max,
                            dst + static_cast<uint32_t>(oc) * plane)) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
  }
  q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
  return V;
}

// The AVR contract uses only an INT32 dot product. Adding the bias in INT64
// before saturation preserves the original quantized rounding and clamping.
// All AVR FCN entry paths must validate q_fcn_accumulator_supported() first.
static bool q_fcn_accumulator_supported(const NoodleTensor *input,
                                        size_t count, int32_t input_zp) {
#if defined(__AVR__)
  // Every signed INT8 product has magnitude <= 255 * 128 = 32640 when
  // the input zero-point is signed INT8. An INT32 dot product is safe up to
  // floor(INT32_MAX / 32640) = 65793 elements.
  // Independently verify the FULL tensor dimensions before trusting size_t:
  // size_t is only 16 bits on ATmega328P, so noodle_tensor_size() may wrap.
  if (!input || !input->C || !input->W ||
      (input->rank != NOODLE_TENSOR_1D && input->rank != NOODLE_TENSOR_2D) ||
      input_zp < -128 || input_zp > 127) return false;
  const uint32_t max_dot = static_cast<uint32_t>(INT32_MAX) / 32640UL;
  const uint32_t max_count = static_cast<uint32_t>(SIZE_MAX) < max_dot
                                 ? static_cast<uint32_t>(SIZE_MAX) : max_dot;
  uint32_t n = static_cast<uint32_t>(input->C) * input->W;
  if (input->rank == NOODLE_TENSOR_2D) {
    if (n > max_count / input->W) return false;
    n *= input->W;
  }
  return n <= max_count && static_cast<size_t>(n) == count;
#else
  (void)input;
  (void)count;
  (void)input_zp;
  return true;  // Other targets retain their existing INT64 behavior.
#endif
}

static int32_t q_fcn_dot(const int8_t *input, const int8_t *weight,
                         size_t I, int32_t input_zp, int32_t bias) {
#if defined(__AVR__)
  int32_t dot = 0;
  for (size_t i = 0; i < I; ++i)
    dot += (static_cast<int32_t>(input[i]) - input_zp) * weight[i];
  return q_sat_i32(static_cast<int64_t>(dot) + bias);
#else
  int64_t acc = bias;
  for (size_t i = 0; i < I; ++i)
    acc += static_cast<int64_t>(static_cast<int32_t>(input[i]) - input_zp) * weight[i];
  return q_sat_i32(acc);
#endif
}


// FCN streaming: bounded INT8 weight blocks, for both SRAM-input and
// file-backed input tensors. A file-backed input needs two blocks (X and W).
// Size is configured in noodle_config.h as NOODLE_INT8_FCN_BLOCK.
struct QI8FCWeights {
  const int8_t *ram = nullptr;
  const int32_t *bias = nullptr;
  const int32_t *mult = nullptr;
  const int32_t *shift = nullptr;
  NDL_File wf, bf, mf, sf;
  bool has_bias_file = false;
  bool progmem = false;
  uint32_t far_weight = 0, far_bias = 0, far_mult = 0, far_shift = 0;
  ~QI8FCWeights() {
    if (wf) wf.close(); if (bf) bf.close();
    if (mf) mf.close(); if (sf) sf.close();
  }
  bool parameters(uint16_t oc, int32_t *b, int32_t *m, int32_t *sh) {
    if (wf) {
      *b = has_bias_file ? noodle_read_i32(bf) : 0;
      *m = noodle_read_i32(mf);
      *sh = noodle_read_i32(sf);
    } else if (progmem) {
#if defined(__AVR__)
      *b = far_bias ? noodle_pgm_i32_address(far_bias + uint32_t(oc) * 4u) : 0;
      *m = noodle_pgm_i32_address(far_mult + uint32_t(oc) * 4u);
      *sh = noodle_pgm_i32_address(far_shift + uint32_t(oc) * 4u);
#else
      (void)oc; (void)b; (void)m; (void)sh;
      return false;
#endif
    } else {
      *b = bias ? bias[oc] : 0;
      *m = mult[oc];
      *sh = shift[oc];
    }
    return true;
  }
  bool values(uint16_t oc, size_t I, size_t offset, size_t count, int8_t *dst) {
    if (wf) return q_read_i8_block(wf, dst, count) == count;
    if (progmem) {
#if defined(__AVR__)
      const uint32_t base = far_weight + static_cast<uint32_t>(oc) * I + offset;
      for (size_t j = 0; j < count; ++j)
        dst[j] = noodle_pgm_i8_address(base + static_cast<uint32_t>(j));
      return true;
#else
      return false;
#endif
    }
    memcpy(dst, ram + static_cast<size_t>(oc) * I + offset, count);
    return true;
  }
};
static bool q_fc_weights_init(const FCNMem &l, QI8FCWeights *s) {
  if (!l.weight || !l.multiplier || !l.shift) return false;
  s->ram = l.weight; s->bias = l.bias;
  s->mult = l.multiplier; s->shift = l.shift;
  return true;
}
static bool q_fc_weights_init(const FCNFile &l, QI8FCWeights *s) {
  if (!q_open_required(l.weight_fn, &s->wf)) { NDL_FCN_STAGE(31); return false; }
  if (!q_open_required(l.multiplier_fn, &s->mf)) { NDL_FCN_STAGE(32); return false; }
  if (!q_open_required(l.shift_fn, &s->sf)) { NDL_FCN_STAGE(33); return false; }
  if (l.bias_fn) {
    s->has_bias_file = q_open_required(l.bias_fn, &s->bf);
    if (!s->has_bias_file) { NDL_FCN_STAGE(34); return false; }
  }
  return true;
}
static bool q_fc_weights_init(const FCNProgmem &l, QI8FCWeights *s) {
#if defined(__AVR__)
  if (!l.weight_far || !l.multiplier_far || !l.shift_far) return false;
  s->progmem = true; s->far_weight = l.weight_far;
  s->far_bias = l.bias_far; s->far_mult = l.multiplier_far;
  s->far_shift = l.shift_far;
  return true;
#else
  (void)l; (void)s;
  return false;
#endif
}

template <typename Layer>
#if defined(__AVR__)
__attribute__((noinline))
#endif
static uint16_t q_fcn_file_input(NoodleTensor *input, const Layer &l,
                                  NoodleTensor *output) {
  if (!input || !output || input == output || !input->file_path ||
      !l.O || !q_input_matches(input, l.input_scale, l.input_zero_point) ||
      (output->file_path && strcmp(input->file_path, output->file_path) == 0))
    return 0;
  const size_t I = noodle_tensor_size(input);
  if (!I || !q_fcn_accumulator_supported(input, I, l.input_zero_point)) return 0;
  QI8FCWeights weights;
  if (!q_fc_weights_init(l, &weights)) return 0;
  QI8ActivationWriter writer(output, l.O, 1, l.output_scale,
                            l.output_zero_point, true);
  if (!writer.valid) return 0;
  int8_t xs[NOODLE_INT8_FCN_BLOCK], ws[NOODLE_INT8_FCN_BLOCK];
  for (uint16_t o = 0; o < l.O; ++o) {
    NDL_File file = noodle_fs_open_read(input->file_path);
    if (!file || file.size() != I) { if (file) file.close(); return 0; }
#if defined(__AVR__)
    int32_t acc = 0;
#else
    int64_t acc = 0;
#endif
    for (size_t base = 0; base < I; base += NOODLE_INT8_FCN_BLOCK) {
      const size_t n = (I - base < NOODLE_INT8_FCN_BLOCK) ? I - base
                                                         : NOODLE_INT8_FCN_BLOCK;
      if (noodle_read_raw(file, xs, n) != n ||
          !weights.values(o, I, base, n, ws)) { file.close(); return 0; }
#if defined(__AVR__)
      for (size_t k = 0; k < n; ++k)
        acc += (static_cast<int32_t>(xs[k]) - l.input_zero_point) * ws[k];
#else
      for (size_t k = 0; k < n; ++k)
        acc += (static_cast<int64_t>(xs[k]) - l.input_zero_point) * ws[k];
#endif
    }
    file.close();
    int32_t bias, mult, shift;
    if (!weights.parameters(o, &bias, &mult, &shift)) return 0;
    const int32_t acc32 = q_sat_i32(static_cast<int64_t>(acc) + bias);
    if (!writer.put(static_cast<int8_t>(q_apply_output(
            acc32, mult, shift, l.output_zero_point,
            l.activation_min, l.activation_max)))) return 0;
  }
  if (!writer.complete(l.O, 1, l.output_scale,
                       l.output_zero_point, true)) return 0;
  return l.O;
}

// Case 1 (RAM input/output + SD parameters): use the original FP32 FCN's
// two shared file-handle pattern instead of four stack-local NDL_File members.
// Keep the weight stream open in fw; reopen the metadata stream in fb only
// while extracting one neuron's int32 value from an existing .bin file.
// This preserves all eight parameter files and unchanged INT8 arithmetic.
// It intentionally trades extra metadata opens/seeks for lower peak SRAM.
static bool q_fcn_case1_read_i32_at(const char *filename,
                                   uint16_t neuron, int32_t *value) {
  if (!filename || !value) return false;
  // Only fw (weights) and fb (one metadata file) may be open together.
  fb = noodle_fs_open_read(filename);
  if (!fb) return false;
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  const uint32_t off = static_cast<uint32_t>(neuron) * sizeof(int32_t);
#if defined(NOODLE_USE_SDFAT)
  const bool seek_ok = fb.seekSet(off);
#else
  const bool seek_ok = fb.seek(off);
#endif
  const bool ok = seek_ok &&
                  noodle_read_raw(fb, value, sizeof(int32_t)) == sizeof(int32_t);
#else
  // Text backend: scalar offsets are lines, not fixed-size byte positions.
  bool ok = true;
  for (uint16_t i = 0; i <= neuron; ++i) {
    if (!fb.available()) { ok = false; break; }
    *value = noodle_read_i32(fb);
  }
#endif
  fb.close();
  return ok;
}

// Case 1: input, hidden and output activations stay in SRAM.
// Keep SD weights and per-neuron metadata sequential and use exactly two
// FP32-style shared file handles. The packed file holds three LE int32s.
static int32_t q_case1_le_i32(const uint8_t *p) {
  const uint32_t u = static_cast<uint32_t>(p[0]) |
                     (static_cast<uint32_t>(p[1]) << 8) |
                     (static_cast<uint32_t>(p[2]) << 16) |
                     (static_cast<uint32_t>(p[3]) << 24);
  // Avoid relying on implementation-defined uint32_t -> int32_t conversion.
  if (u <= 0x7FFFFFFFUL) return static_cast<int32_t>(u);
  return -1 - static_cast<int32_t>(~u);
}

// Shared block loop: the accumulator type is selected at compile time
// (INT32 on AVR, INT64 where the existing generic path requires it).
template <typename Accumulator>
static inline void q_fcn_accumulate_block(Accumulator *sum,
                                          const int8_t *xs,
                                          const int8_t *ws,
                                          size_t count, int32_t input_zp) {
  for (size_t k = 0; k < count; ++k) {
    const Accumulator x = static_cast<Accumulator>(xs[k]) -
                          static_cast<Accumulator>(input_zp);
    *sum += x * static_cast<Accumulator>(ws[k]);
  }
}

// INT32 FCN dot-product accumulation is mandatory on AVR. The arithmetic
// bound is checked before inference rather than dispatched per neuron.
// For packed SD metadata on AVR, the caller opens the shared fw/fb handles
// in its shallow stack frame before entering the noinline compute kernel.
// AVR + SdFat opens directly into the handles to avoid temporary File objects.
// Other targets retain the original file-opening path.

// Non-AVR SRAM-input / SD-weight FCN: one monomorphic MAC loop for INT32
// and one for the INT64 fallback, avoiding a branch per multiply.
// A whole matrix is NEVER loaded into SRAM: this stages one weight block.
#if !defined(__AVR__)
static bool q_fcn_sd_dot_acc32_safe(size_t count, int32_t input_zp) {
  return input_zp >= -128 && input_zp <= 127 &&
         static_cast<uint64_t>(count) <=
             static_cast<uint64_t>(INT32_MAX) / (255u * 128u);
}

template <typename Accumulator>
static bool q_fcn_sd_dot(NDL_File &weight_file, const int8_t *xs,
                         size_t count, int32_t input_zp,
                         Accumulator *dot) {
  int8_t block[NOODLE_INT8_FCN_BLOCK];
  Accumulator sum = 0;
  for (size_t base = 0; base < count; base += NOODLE_INT8_FCN_BLOCK) {
    const size_t n = (count - base < NOODLE_INT8_FCN_BLOCK)
                         ? count - base : NOODLE_INT8_FCN_BLOCK;
    if (q_read_i8_block(weight_file, block, n) != n) return false;
    for (size_t k = 0; k < n; ++k) {
      const Accumulator x = static_cast<Accumulator>(xs[base + k]) -
                            static_cast<Accumulator>(input_zp);
      sum += x * static_cast<Accumulator>(block[k]);
    }
  }
  *dot = sum;
  return true;
}
#endif

// A separate call frame prevents the large weight staging array from being
// live while the shallow caller opens the SD files on AVR.
#if defined(__AVR__)
__attribute__((noinline))
#endif
static uint16_t q_fcn_ram_to_ram_file_weights(
    NoodleTensor *input, const FCNFile &l, NoodleTensor *output,
    bool preopened) {
  NDL_MEM_CHECK();
  NDL_FCN_STAGE(10);
  const bool packed = l.metadata_fn != nullptr;
#if NOODLE_FILE_FORMAT != NOODLE_FILE_FORMAT_BIN
  if (packed) {
    NDL_FCN_STAGE(13);
    if (preopened) { fb.close(); fw.close(); }
    return 0;
  }
#endif
  // The kernel owns both shared handles only when its caller explicitly
  // preopened them. Close on EVERY early exit, including arena allocation
  // failure, so later FCN invocations start with clean file-handle state.
  const bool already_open = packed && preopened;
  if (!input || !output || input == output || input->file_path ||
      output->file_path || !input->buffer.data || !l.O ||
      !q_input_matches(input, l.input_scale, l.input_zero_point) ||
      !l.weight_fn || (!packed && (!l.multiplier_fn || !l.shift_fn))) {
    NDL_FCN_STAGE(11);
    if (already_open) { fb.close(); fw.close(); }
    return 0;
  }
  const size_t I = noodle_tensor_size(input);
  if (!I || !q_fcn_accumulator_supported(input, I, l.input_zero_point)) {
    NDL_FCN_STAGE(12);
    if (already_open) { fb.close(); fw.close(); }
    return 0;
  }

  // Output storage is usually preallocated in Nano demos, but failures are
  // still possible (invalid quantization, insufficient heap, etc.).
  int8_t *dst = noodle_i8_tensor_require_vector(
      output, l.O, l.output_scale, l.output_zero_point);
  if (!dst) {
    NDL_FCN_STAGE(41);
    if (already_open) { fb.close(); fw.close(); }
    return 0;
  }
  const int8_t *xs = input->buffer.data;
  NDL_FCN_STAGE(20);
  if (!already_open) {
    fw = noodle_fs_open_read(l.weight_fn);
    if (!fw) { NDL_FCN_STAGE(31); return 0; }
    NDL_MEM_CHECK();
  } else if (!fw) {
    NDL_FCN_STAGE(31);
    return 0;
  }
  if (packed) {
    if (!already_open) {
      fb = noodle_fs_open_read(l.metadata_fn);
      if (!fb) { NDL_FCN_STAGE(35); fw.close(); return 0; }
      NDL_MEM_CHECK();
    } else if (!fb) {
      NDL_FCN_STAGE(35); fw.close(); return 0;
    }
    const uint32_t expect = static_cast<uint32_t>(l.O) * 12u;
    if (fb.size() != expect) {
      NDL_FCN_STAGE(36); fb.close(); fw.close(); return 0;
    }
  }
  NDL_MEM_CHECK();
#if defined(__AVR__)
  int8_t ws[NOODLE_INT8_FCN_BLOCK];
#else
  const bool use_acc32 = q_fcn_sd_dot_acc32_safe(I, l.input_zero_point);
#endif
  NDL_FCN_STAGE(50);
  for (uint16_t o = 0; o < l.O; ++o) {
    if (o == 0) NDL_MEM_CHECK();
    NDL_FCN_MARK('N', o);
#if !defined(__AVR__)
    int64_t acc = 0;
    bool success;
    if (use_acc32) {
      int32_t dot = 0;
      success = q_fcn_sd_dot<int32_t>(fw, xs, I, l.input_zero_point, &dot);
      acc = dot;
    } else {
      success = q_fcn_sd_dot<int64_t>(fw, xs, I, l.input_zero_point, &acc);
    }
    if (!success) {
      NDL_FCN_STAGE(51);
      if (packed) fb.close();
      fw.close();
      return 0;
    }
#else
    int32_t acc = 0;
    for (size_t base = 0; base < I; base += NOODLE_INT8_FCN_BLOCK) {
      const size_t n = (I - base < NOODLE_INT8_FCN_BLOCK)
                           ? I - base : NOODLE_INT8_FCN_BLOCK;
      if (q_read_i8_block(fw, ws, n) != n) {
        NDL_FCN_STAGE(51);
        if (packed) fb.close();
        fw.close();
        return 0;
      }
      q_fcn_accumulate_block(&acc, xs + base, ws, n,
                             l.input_zero_point);
      if ((base + n) == I || ((base + n) % 64u) == 0u)
        NDL_FCN_MARK('W', o);
    }
#endif  // __AVR__: INT32-only dot product

    int32_t bias = 0, mult = 0, shift = 0;
    if (packed) {
      uint8_t record[12];
      NDL_FCN_MARK('p', o);
      if (noodle_read_raw(fb, record, sizeof(record)) != sizeof(record)) {
        NDL_FCN_STAGE(52); fb.close(); fw.close(); return 0;
      }
      bias = q_case1_le_i32(record);
      mult = q_case1_le_i32(record + 4);
      shift = q_case1_le_i32(record + 8);
      NDL_FCN_MARK('P', o);
    } else {
      if (l.bias_fn && !q_fcn_case1_read_i32_at(l.bias_fn, o, &bias)) {
        NDL_FCN_STAGE(34); fw.close(); return 0;
      }
      if (!q_fcn_case1_read_i32_at(l.multiplier_fn, o, &mult)) {
        NDL_FCN_STAGE(32); fw.close(); return 0;
      }
      if (!q_fcn_case1_read_i32_at(l.shift_fn, o, &shift)) {
        NDL_FCN_STAGE(33); fw.close(); return 0;
      }
    }

    // INT64 bias addition happens just once per output, never per MAC.
    const int32_t acc_bias = q_sat_i32(static_cast<int64_t>(acc) + bias);
    dst[o] = static_cast<int8_t>(q_apply_output(
        acc_bias, mult, shift, l.output_zero_point,
        l.activation_min, l.activation_max));
    NDL_FCN_MARK('E', o);

  }

  if (packed) fb.close();
  fw.close();
  NDL_MEM_CHECK();
  NDL_FCN_STAGE(0);
  return l.O;
}

template <typename Layer>
#if defined(__AVR__)
__attribute__((noinline))
#endif
static uint16_t q_fcn_ram_input(NoodleTensor *input, const Layer &l,
                                 NoodleTensor *output) {
  if (!input || !output || input == output || input->file_path ||
      !input->buffer.data || !l.O ||
      !q_input_matches(input, l.input_scale, l.input_zero_point)) return 0;
  const size_t I = noodle_tensor_size(input);
  if (!I || !q_fcn_accumulator_supported(input, I, l.input_zero_point)) return 0;
  QI8FCWeights weights;
  if (!q_fc_weights_init(l, &weights)) return 0;
  QI8ActivationWriter writer(output, l.O, 1, l.output_scale,
                            l.output_zero_point, true);
  if (!writer.valid) return 0;
  int8_t ws[NOODLE_INT8_FCN_BLOCK];
  const int8_t *xs = input->buffer.data;
  for (uint16_t o = 0; o < l.O; ++o) {
#if defined(__AVR__)
    int32_t acc = 0;
#else
    int64_t acc = 0;
#endif
    for (size_t base = 0; base < I; base += NOODLE_INT8_FCN_BLOCK) {
      const size_t n = (I - base < NOODLE_INT8_FCN_BLOCK)
                           ? I - base : NOODLE_INT8_FCN_BLOCK;
      if (!weights.values(o, I, base, n, ws)) return 0;
      q_fcn_accumulate_block(&acc, xs + base, ws, n,
                             l.input_zero_point);
    }
    int32_t bias, mult, shift;
    if (!weights.parameters(o, &bias, &mult, &shift)) return 0;
    if (!writer.put(static_cast<int8_t>(q_apply_output(
            q_sat_i32(static_cast<int64_t>(acc) + bias), mult, shift,
            l.output_zero_point,
            l.activation_min, l.activation_max)))) return 0;
  }
  if (!writer.complete(l.O, 1, l.output_scale,
                       l.output_zero_point, true)) return 0;
  return l.O;
}

uint16_t noodle_i8_fcn(NoodleTensor *input, const FCNMem &layer,
                     NoodleTensor *output) {
  if (input && input->file_path)
    return q_fcn_file_input(input, layer, output);
  if (!input || !output || input == output || !input->buffer.data ||
      !layer.weight || !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const size_t I = noodle_tensor_size(input);
  if (!I || !q_fcn_accumulator_supported(input, I, layer.input_zero_point)) return 0;
  int8_t *dst = noodle_i8_tensor_require_vector(output, layer.O,
                                               layer.output_scale,
                                               layer.output_zero_point);
  if (!dst) return 0;
  const int8_t *src = input->buffer.data;
  for (uint16_t o = 0; o < layer.O; ++o) {
    const int32_t acc = q_fcn_dot(src, layer.weight + static_cast<size_t>(o) * I,
                                  I, layer.input_zero_point,
                                  layer.bias ? layer.bias[o] : 0);
    dst[o] = static_cast<int8_t>(q_apply_output(acc, layer.multiplier[o],
                                                layer.shift[o],
                                                layer.output_zero_point,
                                                layer.activation_min,
                                                layer.activation_max));
  }
  return layer.O;
}

#if defined(__AVR__)
__attribute__((noinline))
#endif
uint16_t noodle_i8_fcn(NoodleTensor *input, const FCNFile &layer,
                       NoodleTensor *output) {
  if (input && input->file_path)
    return q_fcn_file_input(input, layer, output);
  if (output && !output->file_path) {
    bool preopened = false;
#if defined(__AVR__) && NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
    // Packed SD parameters: open both shared handles in the shallower
    // dispatch frame, before the compute kernel's weight block is live.
    // The kernel always owns and closes both handles on return.
    if (input && input != output && !input->file_path && input->buffer.data &&
        layer.weight_fn && layer.metadata_fn && layer.O &&
        q_input_matches(input, layer.input_scale, layer.input_zero_point) &&
        noodle_tensor_size(input) != 0u &&
        q_fcn_accumulator_supported(input, noodle_tensor_size(input),
                                    layer.input_zero_point)) {
#if defined(NOODLE_USE_SDFAT)
      // SdFat File.open() avoids a by-value File returned from FS.open().
      if (!fw.open(layer.weight_fn, O_RDONLY)) {
        NDL_FCN_STAGE(31); return 0;
      }
#else
      fw = noodle_fs_open_read(layer.weight_fn);
      if (!fw) { NDL_FCN_STAGE(31); return 0; }
#endif
      NDL_MEM_CHECK();
#if defined(NOODLE_USE_SDFAT)
      if (!fb.open(layer.metadata_fn, O_RDONLY)) {
        NDL_FCN_STAGE(35); fb.close(); fw.close(); return 0;
      }
#else
      fb = noodle_fs_open_read(layer.metadata_fn);
      if (!fb) { NDL_FCN_STAGE(35); fw.close(); return 0; }
#endif
      NDL_MEM_CHECK();
      preopened = true;
    }
#endif
    return q_fcn_ram_to_ram_file_weights(input, layer, output, preopened);
  }
  return q_fcn_ram_input(input, layer, output);
}

uint16_t noodle_i8_conv2d(NoodleTensor *input, const ConvProgmem &layer,
                          NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, g.V,
                                            layer.output_scale,
                                            layer.output_zero_point);
  if (!dst) return 0;
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K * layer.K;
  int8_t *filter = q_filter_require(filter_size);
  if (!filter) return 0;
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    const uint32_t base = static_cast<uint32_t>(oc) * filter_size;
    for (size_t i = 0; i < filter_size; ++i) {
      filter[i] = noodle_pgm_i8(layer.weight, base + static_cast<uint32_t>(i));
    }
    const int32_t bias = layer.bias ? noodle_pgm_i32(layer.bias, oc) : 0;
    const int32_t mult = noodle_pgm_i32(layer.multiplier, oc);
    const int32_t shift = noodle_pgm_i32(layer.shift, oc);
    if (!q_conv2d_filter(input, g.V, g.P0, filter, bias, mult, shift,
                         layer.K, layer.S, layer.input_zero_point,
                         layer.output_zero_point, layer.activation_min,
                         layer.activation_max,
                         dst + static_cast<uint32_t>(oc) * out_plane)) return 0;
  }
  return g.V;
}

uint16_t noodle_i8_dwconv2d(NoodleTensor *input, const ConvProgmem &layer,
                            NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || layer.depth_multiplier == 0 ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint32_t expected_o = static_cast<uint32_t>(input->C) * layer.depth_multiplier;
  const uint16_t O = layer.O ? layer.O : static_cast<uint16_t>(expected_o);
  if (expected_o > 65535u || O != expected_o) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, O, g.V,
                                            layer.output_scale,
                                            layer.output_zero_point);
  if (!dst) return 0;
  const size_t ksize = static_cast<size_t>(layer.K) * layer.K;
  int8_t *kernel = q_filter_require(ksize);
  if (!kernel) return 0;
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < O; ++oc) {
    const uint32_t base = static_cast<uint32_t>(oc) * ksize;
    for (size_t i = 0; i < ksize; ++i) {
      kernel[i] = noodle_pgm_i8(layer.weight, base + static_cast<uint32_t>(i));
    }
    const uint16_t ic = oc / layer.depth_multiplier;
    const int32_t bias = layer.bias ? noodle_pgm_i32(layer.bias, oc) : 0;
    const int32_t mult = noodle_pgm_i32(layer.multiplier, oc);
    const int32_t shift = noodle_pgm_i32(layer.shift, oc);
    int8_t *out = dst + static_cast<uint32_t>(oc) * out_plane;
    for (uint16_t oy = 0; oy < g.V; ++oy) {
      for (uint16_t ox = 0; ox < g.V; ++ox) {
        const int32_t acc = q_dw_at(input, ic, kernel, layer.K, oy, ox,
                                    g.P0, layer.S, layer.input_zero_point, bias);
        out[static_cast<uint32_t>(oy) * g.V + ox] = static_cast<int8_t>(
            q_apply_output(acc, mult, shift, layer.output_zero_point,
                           layer.activation_min, layer.activation_max));
      }
    }
  }
  return g.V;
}

uint16_t noodle_i8_fcn(NoodleTensor *input, const FCNProgmem &layer,
                       NoodleTensor *output) {
  return input && input->file_path
      ? q_fcn_file_input(input, layer, output)
      : q_fcn_ram_input(input, layer, output);
}

#endif  // NOODLE_USE_INT8
