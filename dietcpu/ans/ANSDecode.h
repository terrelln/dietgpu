#pragma once

#include <stddef.h>
#include <stdint.h>

namespace dietcpu {

size_t ansDecode(void* dst, size_t dstCapacity, void const* src, size_t srcSize);

}