#pragma once

#include <vector>

#include "ANSUtils.h"

namespace dietcpu {

// [0, 12)  = pdf
// [12, 24) = cdf
// [24, 29) = shift
// [29, 32) = unused
// [32, 64) = div_m1
using ANSTable = uint64_t;

std::vector<uint32_t> ansHistogram(ANSDecodedT const *src, size_t srcSize);

std::vector<ANSTable> ansCalcWeights(int probBits, uint32_t const *histogram, size_t srcSize);
} // namespace dietcpu