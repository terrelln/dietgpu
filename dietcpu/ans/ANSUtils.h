#pragma once

#include <assert.h>

#include "../utils/StaticUtils.h"

namespace dietcpu {

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
}