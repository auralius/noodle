/** Shared backend objects and file handles for both numerical modes. */
#include "noodle_internal.h"

#if defined(NOODLE_USE_SDFAT)
SdFat NOODLE_FS;
#endif

NDL_File fw, fb, fo, fi;
void *temp_buff1 = NULL;
void *temp_buff2 = NULL;
size_t temp_buff1_capacity = 0;
size_t temp_buff2_capacity = 0;
