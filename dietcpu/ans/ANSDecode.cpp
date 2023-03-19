#include "ANSDecode.h"

#include <array>
#include <immintrin.h>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "ANSUtils.h"

namespace dietcpu {
namespace {
constexpr auto constructReadPermute() {
  std::array<std::array<uint32_t, 8>, 256> permute;
  for (int i = 0; i < permute.size(); ++i) {
    uint32_t remainder = 8;
    for (int j = permute[i].size() - 1; j >= 0; --j) {
      bool const bitSet = (i & (1 << j)) != 0;
      if (bitSet) {
        remainder -= 1;
        permute[i][j] = remainder;
      } else {
        permute[i][j] = 0;
      }
    }
  }
  return permute;
}

constexpr auto __attribute__((__aligned__(32))) kReadPermute =
    constructReadPermute();

VectorAVX2 readPermute(int readM) {
  return VectorAVX2(VectorAVX2::Aligned{},
                    (__m256i const *)kReadPermute[readM].data());
}

size_t constexpr kBlockSize = kDefaultBlockSize;

using TableT = uint32_t;

// We are limited to 11 bits of probability resolution
// (worst case, prec = 12, pdf == 2^12, single symbol. 2^12 cannot be
// represented in 12 bits)
inline TableT packDecodeLookup(uint32_t sym, uint32_t pdf, uint32_t cdf) {
  static_assert(sizeof(ANSDecodedT) == 1, "");
  // [31:20] cdf
  // [19:8] pdf
  // [7:0] symbol
  return (cdf << 20) | (pdf << 8) | sym;
}

std::vector<TableT> ansDecodeTable(ANSCoalescedHeader const &header) {
  auto const probBits = header.getProbBits();
  auto probs = header.getSymbolProbs();
  uint32_t cdf = 0;

  std::vector<TableT> table;
  table.resize(1 << probBits);

  for (size_t i = 0; i < kNumSymbols; ++i) {
    uint32_t pdf = probs[i];

    uint32_t const begin = cdf;
    uint32_t const end = begin + pdf;

    for (size_t j = begin; j < end; ++j) {
      table[j] = packDecodeLookup(i, pdf, j - begin);
    }

    cdf = end;
  }
  assert(cdf == table.size());

  return table;
}

template <size_t kProbBits> class ANSTableAVX2 : public VectorAVX2 {
public:
  explicit ANSTableAVX2() {}
  explicit ANSTableAVX2(TableT const *table, VectorAVX2 indices)
      : VectorAVX2(indices.gather32((int const *)table)) {}

  /// 2. Get the symbol from the table.
  VectorAVX2 symbol() const { return *this & VectorAVX2(0xFF); }

  VectorAVX2 pdf() const { return (*this >> 8) & VectorAVX2(0xFFF); }

  VectorAVX2 cdf() const { return *this >> 20; }

private:
  __m256i stateX8_;
};

template <size_t kProbBits> class ANSStateAVX2 : public VectorAVX2 {
private:
  explicit ANSStateAVX2(VectorAVX2 const &v) : VectorAVX2(v) {}

public:
  explicit ANSStateAVX2() {}

  explicit ANSStateAVX2(ANSStateT const *states)
      : VectorAVX2(VectorAVX2::Unaligned{}, (__m256i_u const *)states) {}

  ANSTableAVX2<kProbBits> lookup(TableT const *table) const {
    auto const indices = *this & VectorAVX2((1u << kProbBits) - 1);
    return ANSTableAVX2<kProbBits>(table, indices);
  }

  int update(ANSTableAVX2<kProbBits> const &table, ANSEncodedT const *in) {
    auto dataV = VectorAVX2::loadWordForEachState(in);

    auto pdf = table.pdf();
    auto cdf = table.cdf();

    auto state = (pdf * (*this >> kProbBits)) + cdf;

    auto const readV = state < VectorAVX2(kANSMinState);
    auto const readM = readV.mask();
    auto const permV = readPermute(readM);
    dataV = dataV.permute8x32(permV);
    auto const nextV = (state << kANSEncodedBits) + dataV;

    *this = ANSStateAVX2(readV.blend(state, nextV));

    return _mm_popcnt_u32(readM);
  }
};

void write(ANSDecodedT *out, VectorAVX2 symbols0V, VectorAVX2 symbols1V) {
  VectorAVX2 symbolsV = _mm256_packus_epi32(*symbols0V, *symbols1V);
  symbolsV = _mm256_permute4x64_epi64(*symbolsV, 0xD8);
  symbolsV = _mm256_packus_epi16(*symbolsV, *symbolsV);

  uint64_t const x0 = _mm256_extract_epi64(*symbolsV, 0);
  uint64_t const x1 = _mm256_extract_epi64(*symbolsV, 2);
  memcpy(out + 0, &x0, 8);
  memcpy(out + 8, &x1, 8);
}

template <size_t kProbBits>
void ansDecodeFullBlock(ANSStateT const *states, ANSDecodedT *out,
                        ANSEncodedT const *in, size_t inSize,
                        TableT const *table) {
  in += inSize;

  std::array<ANSStateAVX2<kProbBits>, 4> statesV;
  for (size_t i = 0; i < statesV.size(); ++i) {
    statesV[i] = ANSStateAVX2<kProbBits>(states + i * 8);
  }

  std::array<ANSTableAVX2<kProbBits>, 4> tablesV;
  for (size_t i = 0; i < statesV.size(); ++i) {
    tablesV[i] = statesV[i].lookup(table);
  }

  for (int i = kBlockSize - kWarpSize; i >= 0; i -= kWarpSize) {
    for (int s = statesV.size() - 2; s >= 0; s -= 2) {
      auto const symbols0V = tablesV[s + 0].symbol();
      auto const symbols1V = tablesV[s + 1].symbol();

      write(out + i + 8 * s, symbols0V, symbols1V);

      for (int t = 1; t >= 0; --t) {
        auto &stateV = statesV[s + t];
        auto &tableV = tablesV[s + t];

        // Update the state
        in -= stateV.update(tableV, in);

        // Reload the table
        tableV = stateV.lookup(table);
      }
    }
  }
}
} // namespace

size_t ansDecode(void *dst, size_t dstCapacity, void const *src,
                 size_t srcSize) {
  auto const &header = *(const ANSCoalescedHeader *)src;

  header.checkMagicAndVersion();
  auto const numBlocks = header.getNumBlocks();
  auto const totalUncompressedWords = header.getTotalUncompressedWords();
  auto const probBits = header.getProbBits();

  if (totalUncompressedWords == 0) {
    return 0;
  }

  static_assert(sizeof(ANSDecodedT) == 1, "");
  if (dstCapacity < totalUncompressedWords) {
    throw DstCapacityTooSmallError();
  }
  if (totalUncompressedWords != numBlocks * kBlockSize) {
    throw PartialBlockError();
  }

  auto out = (ANSDecodedT *)dst;

  auto table = ansDecodeTable(header);

  for (size_t block = 0; block < numBlocks; ++block) {
    // Load state
    ANSWarpState const &states = header.getWarpStates()[block];

    // Load per-block size data
    auto blockWords = header.getBlockWords(numBlocks)[block];
    uint32_t uncompressedWords = (blockWords.x >> 16);
    uint32_t compressedWords = (blockWords.x & 0xffff);
    uint32_t blockCompressedWordStart = blockWords.y;

    if (uncompressedWords != kBlockSize) {
      throw PartialBlockError();
    }

    // Get block addresses for encoded/decoded data
    auto blockDataIn =
        header.getBlockDataStart(numBlocks) + blockCompressedWordStart;

    switch (probBits) {
    case 9:
      ansDecodeFullBlock<9>(states.warpState, out + block * kBlockSize,
                            blockDataIn, compressedWords, table.data());
      break;
    case 10:
      ansDecodeFullBlock<10>(states.warpState, out + block * kBlockSize,
                             blockDataIn, compressedWords, table.data());
      break;
    case 11:
      ansDecodeFullBlock<11>(states.warpState, out + block * kBlockSize,
                             blockDataIn, compressedWords, table.data());
      break;
    default:
      throw UnsupportedProbBitsError();
    }
  }
  return totalUncompressedWords;
}
} // namespace dietcpu