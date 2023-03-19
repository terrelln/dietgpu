#pragma once

#include <stddef.h>
#include <stdint.h>

#include "ANSStatistics.h"

namespace dietcpu {
size_t ansEncode(void *dst, size_t dstCapacity, void const *src, size_t srcSize,
                 int probBits);
size_t ansEncode(void *dst, size_t dstCapacity, void const *src, size_t srcSize,
                 int probBits, ANSTable const *table);
} // namespace dietcpu