#include <stdio.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"

// Try one real ESP-DL header found in managed_components
// Example only; adjust to actual header name from find command.
// #include "dl_model_base.hpp"

static const char *TAG = "espdl_test";

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "BOOT ESP-DL test project");

    ESP_LOGI(TAG, "free_internal=%u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    ESP_LOGI(TAG, "free_psram=%u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "largest_psram=%u",
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
}