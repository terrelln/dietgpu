#include "ANSStatistics.h"

#include <algorithm>
#include <array>

#include "ANSUtils.h"

namespace dietcpu {
std::vector<uint32_t> ansHistogram(ANSDecodedT const *src, size_t srcSize) {
  std::vector<uint32_t> histogram;
  histogram.resize(kNumSymbols, 0);
  for (size_t i = 0; i < srcSize; ++i) {
    ++histogram[src[i]];
  }
  return histogram;
}

std::vector<ANSTable> ansCalcWeights(int probBits, uint32_t const *histogram,
                                     size_t srcSize) {
  if (srcSize == 0) {
    return {};
  }
  uint32_t const kProbWeight = 1 << probBits;
  std::array<std::pair<uint16_t, uint8_t>, 256> qProb;
  uint32_t qProbSum = 0;

  for (size_t s = 0; s < kNumSymbols; ++s) {
    uint32_t const count = histogram[s];

    uint32_t qP = (kProbWeight * count) / srcSize;
    qP = (count > 0 && qP == 0) ? 1 : qP;

    qProb[s].first = qP;
    qProb[s].second = s;

    qProbSum += qProb[s].first;
  }

  std::sort(qProb.begin(), qProb.end(), std::greater<>{});

  int diff = (int)kProbWeight - (int)qProbSum;

  if (diff > 0) {
    size_t i = 0;
    while (diff > 0) {
      if (qProb[i].first > 0) {
        qProb[i].first += 1;
        diff -= 1;
      }
      i = (i + 1) % kNumSymbols;
    }
  } else {
    size_t i = kNumSymbols - 1;
    while (diff < 0) {
      if (qProb[i].first > 1) {
        qProb[i].first -= 1;
        diff += 1;
      }
      i = (i - 1) % kNumSymbols;
    }
  }

  std::vector<ANSTable> table;
  table.resize(kNumSymbols);

  qProbSum = 0;
  for (auto const &[pdf, sym] : qProb) {
    table[sym] = pdf;
    qProbSum += pdf;
  }
  assert(qProbSum == kProbWeight);

  uint32_t cdf = 0;
  for (size_t s = 0; s < table.size(); ++s) {
    uint32_t const pdf = table[s];

    if (pdf == 0)
        continue;
    
    uint32_t shift = pdf - 1;
    shift = 32 - (shift == 0 ? 32 : __builtin_clz(shift));

    constexpr uint64_t one = 1;
    uint64_t const divM1 = ((one << 32) * ((one << shift) - pdf)) / pdf + 1;

    assert(pdf < (1 << 12));
    assert(cdf < (1 << 12));
    assert(shift < (1 << 5));
    
    table[s] = (pdf << 0) | (cdf << 12) | (shift << 24) | (divM1 << 32);

    cdf += pdf;
  }

  return table;
}

} // namespace dietcpu