#pragma once

#include <stddef.h>

namespace dietcpu {
size_t ansEncode(void* dst, size_t dstCapacity, void const* src, size_t srcSize, int probBits);
}