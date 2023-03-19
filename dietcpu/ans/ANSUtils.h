#pragma once

#include <assert.h>
#include <immintrin.h>
#include <stdexcept>

#include "../utils/StaticUtils.h"

// Basically a clone of the dietgpu file
// Copied over so I don't have to worry about CUDA

namespace dietcpu {

class DstCapacityTooSmallError : public std::runtime_error {
public:
  DstCapacityTooSmallError() : std::runtime_error("Dst capacity too small!") {}
};

class PartialBlockError : public std::runtime_error {
public:
  PartialBlockError()
      : std::runtime_error("Partial blocks not handled (yet)!") {}
};

class UnsupportedProbBitsError : public std::runtime_error {
public:
  UnsupportedProbBitsError()
      : std::runtime_error("Prob bits must be >= 9 && <= 11!") {}
};

constexpr size_t kWarpSize = 32;

struct uint2 {
  uint32_t x;
  uint32_t y;
};

using ANSStateT = uint32_t;
using ANSEncodedT = uint16_t;
using ANSDecodedT = uint8_t;

struct alignas(16) ANSDecodedTx16 {
  ANSDecodedT x[16];
};

struct alignas(8) ANSDecodedTx8 {
  ANSDecodedT x[8];
};

struct alignas(4) ANSDecodedTx4 {
  ANSDecodedT x[4];
};

constexpr uint32_t kNumSymbols = 1 << (sizeof(ANSDecodedT) * 8);
static_assert(kNumSymbols > 1, "");

// Default block size for compression (in bytes)
constexpr uint32_t kDefaultBlockSize = 4096;

// limit state to 2^31 - 1, so as to prevent addition overflow in the integer
// division via mul and shift by constants
constexpr int kANSStateBits = sizeof(ANSStateT) * 8 - 1;
constexpr int kANSEncodedBits = sizeof(ANSEncodedT) * 8; // out bits
constexpr ANSStateT kANSEncodedMask =
    (ANSStateT(1) << kANSEncodedBits) - ANSStateT(1);

constexpr ANSStateT kANSStartState = ANSStateT(1)
                                     << (kANSStateBits - kANSEncodedBits);
constexpr ANSStateT kANSMinState = ANSStateT(1)
                                   << (kANSStateBits - kANSEncodedBits);

// magic number to verify archive integrity
constexpr uint32_t kANSMagic = 0xd00d;

// current DietGPU version number
constexpr uint32_t kANSVersion = 0x0001;

// Each block of compressed data (either coalesced or uncoalesced) is aligned to
// this number of bytes and has a valid (if not all used) segment with this
// multiple of bytes
constexpr uint32_t kBlockAlignment = 16;

struct ANSWarpState {
  // The ANS state data for this warp
  ANSStateT warpState[kWarpSize];
};

struct alignas(32) ANSCoalescedHeader {
  static uint32_t getCompressedOverhead(uint32_t numBlocks) {
    constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
                                   ? 1
                                   : kBlockAlignment / sizeof(uint2);

    return sizeof(ANSCoalescedHeader) +
           // probs
           sizeof(uint16_t) * kNumSymbols +
           // states
           sizeof(ANSWarpState) * numBlocks +
           // block words
           sizeof(uint2) * roundUp(numBlocks, kAlignment);
  }

  uint32_t getTotalCompressedSize() const {
    return getCompressedOverhead() +
           getTotalCompressedWords() * sizeof(ANSEncodedT);
  }

  uint32_t getCompressedOverhead() const {
    return getCompressedOverhead(getNumBlocks());
  }

  float getCompressionRatio() const {
    return (float)getTotalCompressedSize() /
           (float)getTotalUncompressedWords() * sizeof(ANSDecodedT);
  }

  uint32_t getNumBlocks() const { return numBlocks; }

  void setNumBlocks(uint32_t nb) { numBlocks = nb; }

  void setMagicAndVersion() {
    magicAndVersion = (kANSMagic << 16) | kANSVersion;
  }

  void checkMagicAndVersion() const {
    assert((magicAndVersion >> 16) == kANSMagic);
    assert((magicAndVersion & 0xffffU) == kANSVersion);
  }

  uint32_t getTotalUncompressedWords() const { return totalUncompressedWords; }

  void setTotalUncompressedWords(uint32_t words) {
    totalUncompressedWords = words;
  }

  uint32_t getTotalCompressedWords() const { return totalCompressedWords; }

  void setTotalCompressedWords(uint32_t words) { totalCompressedWords = words; }

  uint32_t getProbBits() const { return options & 0xf; }

  void setProbBits(uint32_t bits) {
    assert(bits <= 0xf);
    options = (options & 0xfffffff0U) | bits;
  }

  bool getUseChecksum() const { return options & 0x10; }

  void setUseChecksum(bool uc) {
    options = (options & 0xffffffef) | (uint32_t(uc) << 4);
  }

  uint32_t getChecksum() const { return checksum; }

  void setChecksum(uint32_t c) { checksum = c; }

  uint16_t *getSymbolProbs() { return (uint16_t *)(this + 1); }

  const uint16_t *getSymbolProbs() const {
    return (const uint16_t *)(this + 1);
  }

  ANSWarpState *getWarpStates() {
    return (ANSWarpState *)(getSymbolProbs() + kNumSymbols);
  }

  const ANSWarpState *getWarpStates() const {
    return (const ANSWarpState *)(getSymbolProbs() + kNumSymbols);
  }

  uint2 *getBlockWords(uint32_t numBlocks) {
    // All of the ANSWarpStates are already kBlockAlignment aligned
    return (uint2 *)(getWarpStates() + numBlocks);
  }

  const uint2 *getBlockWords(uint32_t numBlocks) const {
    // All of the ANSWarpStates are already kBlockAlignment aligned
    return (const uint2 *)(getWarpStates() + numBlocks);
  }

  ANSEncodedT *getBlockDataStart(uint32_t numBlocks) {
    constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
                                   ? 1
                                   : kBlockAlignment / sizeof(uint2);

    return (ANSEncodedT *)(getBlockWords(numBlocks) +
                           roundUp(numBlocks, kAlignment));
  }

  const ANSEncodedT *getBlockDataStart(uint32_t numBlocks) const {
    constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
                                   ? 1
                                   : kBlockAlignment / sizeof(uint2);

    return (const ANSEncodedT *)(getBlockWords(numBlocks) +
                                 roundUp(numBlocks, kAlignment));
  }

  // (16: magic)(16: version)
  uint32_t magicAndVersion;
  uint32_t numBlocks;
  uint32_t totalUncompressedWords;
  uint32_t totalCompressedWords;

  // (27: unused)(1: use checksum)(4: probBits)
  uint32_t options;
  uint32_t checksum;
  uint32_t unused0;
  uint32_t unused1;

  // Data that follows after the header (some of which is variable length):

  // Fixed length array
  // uint16_t probs[kNumSymbols];

  // Variable length array:
  // ANSWarpState states[numBlocks];

  // Per-block information:
  // (uint16: uncompressedWords, uint16: compressedWords)
  // uint32: blockCompressedWordStart
  //
  // Variable length array:
  // uint2 blockWords[roundUp(numBlocks, kBlockAlignment / sizeof(uint2))];

  // Then follows the compressed per-warp/block data for each segment
};

class VectorAVX2 {
public:
  VectorAVX2() {}

  struct Aligned {};
  struct Unaligned {};

  explicit VectorAVX2(Unaligned, __m256i_u const *ptr) {
    v_ = _mm256_loadu_si256(ptr);
  }
  explicit VectorAVX2(Aligned, __m256i const *ptr) {
    v_ = _mm256_load_si256(ptr);
  }

  /* implicit */ VectorAVX2(__m256i v) : v_(v) {}

  explicit VectorAVX2(uint32_t x) : v_(_mm256_set1_epi32(x)) {}

  static VectorAVX2 loadWordForEachState(ANSEncodedT const *end) {
    __m128i const v = _mm_loadu_si128((__m128i_u const *)(end - 8));
    return _mm256_cvtepu16_epi32(v);
  }

  static VectorAVX2 loadWordForEachState(ANSDecodedT const *in) {
    __m128i const v = _mm_loadu_si128((__m128i_u const *)in);
    return _mm256_cvtepu8_epi32(v);
  }

  VectorAVX2 operator&(VectorAVX2 const &o) const {
    return _mm256_and_si256(v_, o.v_);
  }

  VectorAVX2 operator|(VectorAVX2 const &o) const {
    return _mm256_or_si256(v_, o.v_);
  }

  VectorAVX2 operator<<(int shift) const {
    return _mm256_slli_epi32(v_, shift);
  }

  VectorAVX2 operator>>(int shift) const {
    return _mm256_srli_epi32(v_, shift);
  }

  VectorAVX2 operator>>(VectorAVX2 const &o) const {
    return _mm256_srlv_epi32(v_, o.v_);
  }

  VectorAVX2 operator*(VectorAVX2 const &o) const {
    return _mm256_mullo_epi32(v_, o.v_);
  }

  VectorAVX2 operator+(VectorAVX2 const &o) const {
    return _mm256_add_epi32(v_, o.v_);
  }

  VectorAVX2 operator-(VectorAVX2 const &o) const {
    return _mm256_sub_epi32(v_, o.v_);
  }

  VectorAVX2 operator<(VectorAVX2 const &o) const {
    return _mm256_cmpgt_epi32(o.v_, v_);
  }

  VectorAVX2 operator>(VectorAVX2 const &o) const {
    return _mm256_cmpgt_epi32(v_, o.v_);
  }

  int mask() const { return _mm256_movemask_ps((__m256)v_); }

  void storeu(void *p) const { _mm256_storeu_si256((__m256i_u *)p, v_); }

  VectorAVX2 gather32(int32_t const *table, bool emulateGather = false) const {
    if (emulateGather) {
      int indices[8] __attribute__((aligned(32)));
      _mm256_store_si256((__m256i *)indices, v_);
      return _mm256_setr_epi32(table[indices[0]], table[indices[1]],
                               table[indices[2]], table[indices[3]],
                               table[indices[4]], table[indices[5]],
                               table[indices[6]], table[indices[7]]);
    } else {
      return _mm256_i32gather_epi32(table, v_, 4);
    }
  }

  VectorAVX2 gather64(int32_t const *table, bool emulateGather = false) const {
    if (emulateGather) {
      int indices[8] __attribute__((aligned(32)));
      _mm256_store_si256((__m256i *)indices, v_);
      return _mm256_setr_epi32(table[indices[0] * 2], table[indices[1] * 2],
                               table[indices[2] * 2], table[indices[3] * 2],
                               table[indices[4] * 2], table[indices[5] * 2],
                               table[indices[6] * 2], table[indices[7] * 2]);
    } else {
      return _mm256_i32gather_epi32(table, v_, 8);
    }
  }

  VectorAVX2 permute8x32(VectorAVX2 const &p) const {
    return _mm256_permutevar8x32_epi32(v_, p.v_);
  }

  VectorAVX2 blend(VectorAVX2 const &v0, VectorAVX2 const &v1) const {
    return _mm256_blendv_epi8(*v0, *v1, v_);
  }

  VectorAVX2 mulhi(VectorAVX2 const &o) const {
    // Multiply bottom 4 items and top 4 items together.
    VectorAVX2 highMul = _mm256_mul_epu32(_mm256_srli_epi64(**this, 32), _mm256_srli_epi64(*o, 32));
    VectorAVX2 lowMul = _mm256_mul_epu32(**this, *o);

    highMul = highMul & _mm256_set1_epi64x(0xFFFFFFFF00000000ULL);
    lowMul = _mm256_srli_epi64(*lowMul, 32);

    return lowMul | highMul;
  }

  __m256i operator*() const { return v_; }

  void debugPrint(char const *name) const {
#ifndef NDEBUG
    fprintf(stderr, "%10s = [ ", name);
    uint32_t data[8];
    storeu(data);
    for (size_t i = 0; i < 8; ++i) {
      auto const sep = i < 7 ? "," : "";
      fprintf(stderr, "0x%0.8x%s ", data[i], sep);
    }
    fprintf(stderr, "]\n");
#else
    (void)name;
#endif
  }

private:
  __m256i v_;
};
} // namespace dietcpu