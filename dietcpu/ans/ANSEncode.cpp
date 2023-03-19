#include "ANSEncode.h"

#include <array>

#include "ANSStatistics.h"
#include "ANSUtils.h"

namespace dietcpu {
namespace {
constexpr auto constructWritePermute() {
  std::array<std::array<uint32_t, 8>, 256> permute;
  for (int i = 0; i < permute.size(); ++i) {
    for (int j = 0; j < permute[i].size(); ++j) {
      permute[i][j] = 0;
    }
    uint32_t written = 0;
    for (int j = 0; j < permute[i].size(); ++j) {
      bool const bitSet = (i & (1 << j)) != 0;
      if (bitSet) {
        permute[i][written] = j;
        written += 1;
      }
    }
  }
  return permute;
}

constexpr auto __attribute__((__aligned__(32))) kWritePermute =
    constructWritePermute();

VectorAVX2 writePermute(int writeM) {
  (void)writeM;
  return VectorAVX2(VectorAVX2::Aligned{},
                    (__m256i const *)kWritePermute[writeM].data());
}

class ANSTableAVX2 : public VectorAVX2 {
public:
  ANSTableAVX2() {}
  ANSTableAVX2(ANSTable const *table, ANSDecodedT const *symbols) {
    auto symbolsV = VectorAVX2::loadWordForEachState(symbols);
    pdfCdfShift_ = symbolsV.gather64((int32_t const *)table, false);
    divM1_ = symbolsV.gather64((int32_t const *)table + 1, false);
  }

  VectorAVX2 pdf() const { return pdfCdfShift_ & VectorAVX2(0xFFF); }

  VectorAVX2 cdf() const { return (pdfCdfShift_ >> 12) & VectorAVX2(0xFFF); }

  VectorAVX2 shift() const { return pdfCdfShift_ >> 24; }

  VectorAVX2 divM1() const { return divM1_; }

private:
  VectorAVX2 pdfCdfShift_;
  VectorAVX2 divM1_;
};

template <int kProbBits> class ANSStateAVX2 : public VectorAVX2 {
  ANSStateAVX2(VectorAVX2 v) : VectorAVX2(v) {}

public:
  ANSStateAVX2() : VectorAVX2(kANSStartState) {}

  std::pair<int, VectorAVX2> prepareWrite(VectorAVX2 pdf) {
    VectorAVX2 const kStateCheckMulV((1 << (kANSStateBits - kProbBits)));
    auto const writeV = *this > ((pdf * kStateCheckMulV) - VectorAVX2(1));

    auto const writeM = writeV.mask();
    auto const permV = writePermute(writeM);

    auto const dataV = this->permute8x32(permV);

    auto const nextV = *this >> kANSEncodedBits;
    *this = ANSStateAVX2(writeV.blend(*this, nextV));

    return {_mm_popcnt_u32(writeM), dataV & VectorAVX2(kANSEncodedMask)};
  }

  void update(ANSTableAVX2 const &tableV) {
    auto const tV = this->mulhi(tableV.divM1());
    auto const divV = (tV + *this) >> tableV.shift();
    auto const modV = *this - (divV * tableV.pdf());

    *this = divV * VectorAVX2(1u << kProbBits) + modV + tableV.cdf();
  }
};

template <int kProbBits>
ANSEncodedT *write(ANSEncodedT *out, ANSStateAVX2<kProbBits> *statesV,
                   ANSTableAVX2 const *tablesV) {
  auto const [written0, data0V] = statesV[0].prepareWrite(tablesV[0].pdf());
  auto const [written1, data1V] = statesV[1].prepareWrite(tablesV[1].pdf());

  VectorAVX2 dataV = _mm256_packus_epi32(*data0V, *data1V);
  dataV = _mm256_permute4x64_epi64(*dataV, 0xD8);

  __m128i laneV = _mm256_extractf128_si256(*dataV, 0);
  _mm_storeu_si128((__m128i_u *)out, laneV);
  out += written0;

  laneV = _mm256_extractf128_si256(*dataV, 1);
  _mm_storeu_si128((__m128i_u *)out, laneV);
  out += written1;

  return out;
}

template <int kProbBits>
size_t ansEncodeBlockFull(ANSWarpState &states, ANSEncodedT *blockDataOut,
                          ANSDecodedT const *blockDataIn,
                          ANSTable const *table) {
  std::array<ANSStateAVX2<kProbBits>, 4> statesV;
  std::array<ANSTableAVX2, 4> tablesV;

  for (size_t s = 0; s < tablesV.size(); ++s) {
    tablesV[s] = ANSTableAVX2(table, blockDataIn + s * 8);
  }

  ANSEncodedT *out = blockDataOut;

  for (size_t i = 0; i < kDefaultBlockSize; i += kWarpSize) {
    for (size_t s = 0; s < 4; s += 2) {
      out = write(out, statesV.data() + s, tablesV.data() + s);
      for (size_t t = 0; t < 2; ++t) {
        auto &stateV = statesV[s + t];
        auto &tableV = tablesV[s + t];

        stateV.update(tableV);

        if (i < kDefaultBlockSize - kWarpSize) {
          tableV =
              ANSTableAVX2(table, blockDataIn + i + kWarpSize + (s + t) * 8);
        }
      }
    }
  }

  for (size_t s = 0; s < statesV.size(); ++s) {
    statesV[s].storeu(states.warpState + 8 * s);
  }

  return (size_t)(out - blockDataOut);
}
} // namespace

size_t ansEncode(void *dst, size_t dstCapacity, void const *src, size_t srcSize,
                 int probBits) {
  uint32_t const uncompressedWords = srcSize;

  if (srcSize % kDefaultBlockSize != 0) {
    throw PartialBlockError();
  }
  size_t const numBlocks = divUp(srcSize, kDefaultBlockSize);

  // Really rough estimate so I can avoid bounds checking.
  // Clearly this isn't tight or right, just a hack for now.
  if (dstCapacity < srcSize + srcSize / 4 + 2048) {
    throw DstCapacityTooSmallError();
  }

  auto histogram = ansHistogram((ANSDecodedT const *)src, srcSize);
  auto table = ansCalcWeights(probBits, histogram.data(), srcSize);
  auto &header = *(ANSCoalescedHeader *)dst;

  auto const blockDataStart = header.getBlockDataStart(numBlocks);
  auto blockWordsStart = header.getBlockWords(numBlocks);
  auto warpStatesStart = header.getWarpStates();
  auto blockDataOut = blockDataStart;
  for (size_t block = 0; block < numBlocks; ++block) {
    auto const blockDataIn =
        (ANSDecodedT const *)src + block * kDefaultBlockSize;
    size_t compressedBlockWords;
    switch (probBits) {
    case 9:
      compressedBlockWords = ansEncodeBlockFull<9>(
          warpStatesStart[block], blockDataOut, blockDataIn, table.data());
      break;
    case 10:
      compressedBlockWords = ansEncodeBlockFull<10>(
          warpStatesStart[block], blockDataOut, blockDataIn, table.data());
      break;
    case 11:
      compressedBlockWords = ansEncodeBlockFull<11>(
          warpStatesStart[block], blockDataOut, blockDataIn, table.data());
      break;
    default:
      throw UnsupportedProbBitsError();
    }

    assert(compressedBlockWords < (1 << 16));
    blockWordsStart[block].x = (kDefaultBlockSize << 16) | compressedBlockWords;
    blockWordsStart[block].y = (uint32_t)(blockDataOut - blockDataStart);

    blockDataOut += compressedBlockWords;
  }

  uint32_t const totalCompressedWords =
      (uint32_t)(blockDataOut - blockDataStart);

  header.setMagicAndVersion();
  header.setNumBlocks(numBlocks);
  header.setTotalUncompressedWords(uncompressedWords);
  header.setTotalCompressedWords(totalCompressedWords);
  header.setProbBits(probBits);
  header.setUseChecksum(false);

  {
    auto probs = header.getSymbolProbs();
    for (size_t s = 0; s < kNumSymbols; ++s) {
      probs[s] = table[s] & 0xFFF;
    }
  }

  return sizeof(ANSEncodedT) * (blockDataOut - (ANSEncodedT *)dst);
}
} // namespace dietcpu