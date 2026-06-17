// Copyright 2025 Espressif Systems (Shanghai) CO., LTD.
// All rights reserved.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

/**
 * @brief  Convert four characters to FOURCC
 */
#define JPEG_FOURCC_TO_INT(a, b, c, d) ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))

/**
 * @brief  Convert 32-bit FOURCC to string
 */
static inline void jpeg_fourcc_to_str(uint32_t fourcc, char out[5])
{
    for (int i = 0; i < 4; i++) {
        out[i] = (char)((fourcc >> (i * 8)) & 0xFF);
    }
    out[4] = '\0';
}

/**
 * @brief  Macro to convert an FOURCC to a string
 */
#define JPEG_FOURCC_TO_STR(fourcc) ({                                                     \
    static char fourcc_str[5];                                                            \
    fourcc_str[0] = fourcc_str[1] = fourcc_str[2] = fourcc_str[3] = fourcc_str[4] = '\0'; \
    jpeg_fourcc_to_str(fourcc, fourcc_str);                                               \
    fourcc_str;                                                                           \
})

/**
 * @brief  JPEG pixel format
 *
 * @note  Aligned with GMF FourCC definition for audio video codecs and formats
 *        Detailed info refer to `https://github.com/espressif/esp-gmf/blob/main/gmf_core/helpers/include/esp_fourcc.h`
 */
typedef enum {
    JPEG_PIXEL_FORMAT_GRAY       = JPEG_FOURCC_TO_INT('G', 'R', 'E', 'Y'),  /*!< Grayscale. 1-byte luminance component stored in memory for each pixel.
                                                                                 Encoder supported. Decoder un-supported. */
    JPEG_PIXEL_FORMAT_RGB888     = JPEG_FOURCC_TO_INT('R', 'G', 'B', '3'),  /*!< RGB888. 3-bytes red, green and blue components stored in memory from low to high address for each pixel.
                                                                                 Encoder supported. Decoder supported. */
    JPEG_PIXEL_FORMAT_RGBA       = JPEG_FOURCC_TO_INT('R', 'A', '2', '4'),  /*!< RGBA. 4-bytes red, green, blue and alpha components stored in memory from low to high address for each pixel.
                                                                                 Encoder supported. Decoder un-supported. */
    JPEG_PIXEL_FORMAT_YCbYCr     = JPEG_FOURCC_TO_INT('Y', 'U', 'Y', 'V'),  /*!< YCbYCr (belongs to packed yuv422). 4-bytes Y, Cb, Y and Cr components stored in memory from low to high address for each 2 pixels.
                                                                                 Encoder supported. Decoder un-supported. */
    JPEG_PIXEL_FORMAT_YCbY2YCrY2 = JPEG_FOURCC_TO_INT('Y', 'U', 'Y', '2'),  /*!< YCbY2YCrY2 (belongs to packed yuv420). 12-bytes Y, Cb, Y, Y, Cb, Y, Y, Cr, Y, Y, Cr and Y components stored in memory from low to high address for each 8 pixels.
                                                                                 Encoder supported. Decoder un-supported. */
    JPEG_PIXEL_FORMAT_RGB565_BE  = JPEG_FOURCC_TO_INT('R', 'G', 'B', 'B'),  /*!< RGB565. 2-bytes RGB565 big-endian data stored in memory for each pixel.
                                                                                 Encoder supported. Decoder supported. */
    JPEG_PIXEL_FORMAT_RGB565_LE  = JPEG_FOURCC_TO_INT('R', 'G', 'B', 'L'),  /*!< RGB565. 2-bytes RGB565 little-endian data stored in memory for each pixel.
                                                                                 Encoder supported. Decoder supported. */
    JPEG_PIXEL_FORMAT_CbYCrY     = JPEG_FOURCC_TO_INT('U', 'Y', 'V', 'Y'),  /*!< CbYCrY (belongs to packed yuv422). 4-bytes Cb, Y, Cr and Y components stored in memory from low to high address for each 2 pixels.
                                                                                 Encoder supported. Decoder supported. */
} jpeg_pixel_format_t;

/**
 * @brief  JPEG error code
 */
typedef enum {
    JPEG_ERR_OK            = 0,   /*!< Succeeded */
    JPEG_ERR_FAIL          = -1,  /*!< Device error or wrong termination of input stream */
    JPEG_ERR_NO_MEM        = -2,  /*!< Insufficient memory for the image */
    JPEG_ERR_NO_MORE_DATA  = -3,  /*!< Input data is not enough */
    JPEG_ERR_INVALID_PARAM = -4,  /*!< Parameter error */
    JPEG_ERR_BAD_DATA      = -5,  /*!< Data format error (may be damaged data) */
    JPEG_ERR_UNSUPPORT_FMT = -6,  /*!< Right format but not supported */
    JPEG_ERR_UNSUPPORT_STD = -7,  /*!< Not supported JPEG standard */
} jpeg_error_t;

/**
 * @brief  JPEG chrominance subsampling type
 */
typedef enum {
    JPEG_SUBSAMPLE_GRAY = 0,  /*!< Grayscale, no chrominance components */
    JPEG_SUBSAMPLE_444  = 1,  /*!< 4:4:4 chrominance subsampling, one chroma component for every pixel in the source image */
    JPEG_SUBSAMPLE_422  = 2,  /*!< 4:2:2 chrominance subsampling, one chroma component for every 2x1 block of pixels in the source image */
    JPEG_SUBSAMPLE_420  = 3,  /*!< 4:2:0 chrominance subsampling, one chroma component for every 2x2 block of pixels in the source image */
} jpeg_subsampling_t;

/**
 * @brief  JPEG rotation type
 */
typedef enum {
    JPEG_ROTATE_0D   = 0,  /*!< Source image rotates clockwise 0 degree before encoding */
    JPEG_ROTATE_90D  = 1,  /*!< Source image rotates clockwise 90 degree before encoding */
    JPEG_ROTATE_180D = 2,  /*!< Source image rotates clockwise 180 degree before encoding */
    JPEG_ROTATE_270D = 3,  /*!< Source image rotates clockwise 270 degree before encoding */
} jpeg_rotate_t;

/**
 * @brief  JPEG resize resolution parameter
 */
typedef struct {
    uint16_t  width;   /*!< Image width */
    uint16_t  height;  /*!< Image height */
} jpeg_resolution_t;

/**
 * @brief  Allocate buffer. The buffer address will be aligned.
 *
 * @param[in]  size     Allocate buffer size
 * @param[in]  aligned  Aligned byte
 *
 * @return
 *       - NULL    Failed
 *       - Others  Allocated buffer address
 */
void *jpeg_calloc_align(size_t size, int aligned);

/**
 * @brief  Free buffer. The buffer address come from `jpeg_calloc_align`
 *
 * @param[in]  buf  Buffer address
 */
void jpeg_free_align(void *data);

#ifdef __cplusplus
}
#endif  /* __cplusplus */
