#include <stdio.h>
#include <string>
#include <vector>

#include "../fse/fse.h"
#include "ANSDecode.h"
#include "ANSEncode.h"
#include "ANSStatistics.h"

static int help(char const *prog) {
  fprintf(stderr, "USAGE: %s (f)(e9|e10|e11|d) RUNS INFILE [OUTFILE]\n\n",
          prog);
  fprintf(stderr, "(e)ncodes or (d)ecodes INFILE RUNS times and optionally "
                  "writes to OUTFILE\n");
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 4 || argc > 5) {
    return help(argv[0]);
  }

  char const *modeArg = argv[1];
  char const *runsArg = argv[2];
  char const *infileArg = argv[3];
  char const *outfileArg = argc == 5 ? argv[4] : nullptr;

  auto const runs = std::stoull(runsArg);

  if (runs == 0) {
    return help(argv[0]);
  }

  FILE *f = fopen(infileArg, "rb");

  std::vector<uint8_t> data;
  data.resize(100 * 1024 * 1024);
  data.resize(fread(data.data(), 1, data.size(), f));
  fclose(f);

  std::vector<uint8_t> coded;
  coded.resize(100 * 1024 * 1024);

  size_t codedSize;
  if (modeArg[0] == 'f') {
    ++modeArg;
    if (modeArg[0] == 'e') {
      int const probBits = std::stoi(modeArg + 1);
      auto histogram = dietcpu::ansHistogram(data.data(), data.size());
      short norm[256];
      size_t const tableLog = FSE_normalizeCount(
          norm, probBits, histogram.data(), data.size(), 255);
      if (FSE_isError(tableLog)) {
        return 2;
      }
      auto table = FSE_createCTable(255, tableLog);
      if (FSE_isError(FSE_buildCTable(table, norm, 255, tableLog))) {
        return 3;
      }
      size_t const headerSize =
          FSE_writeNCount(coded.data(), coded.size(), norm, 255, tableLog);
      if (FSE_isError(headerSize)) {
        return 4;
      }
      for (size_t i = 0; i < runs; ++i) {

        codedSize = FSE_compress_usingCTable(coded.data() + headerSize,
                                             coded.size() - headerSize,
                                             data.data(), data.size(), table);
      }
      if (FSE_isError(codedSize)) {
        return 5;
      }
      FSE_freeCTable(table);
      codedSize += headerSize;
    } else if (modeArg == std::string("d")) {
      for (size_t i = 0; i < runs; ++i) {
        codedSize = FSE_decompress(coded.data(), coded.size(), data.data(),
                                   data.size());
      if (FSE_isError(codedSize)) {
        return 6;
      }
      }
    } else {
      return help(argv[0]);
    }
  } else {
    if (modeArg[0] == 'e') {
      int const probBits = std::stoi(modeArg + 1);
      auto table = dietcpu::ansBuildTable(data.data(), data.size(), probBits);
      for (size_t i = 0; i < runs; ++i) {
        codedSize = dietcpu::ansEncode(coded.data(), coded.size(), data.data(),
                                       data.size(), probBits, table.data());
      }
    } else if (modeArg == std::string("d")) {
      for (size_t i = 0; i < runs; ++i) {
        codedSize = dietcpu::ansDecode(coded.data(), coded.size(), data.data(),
                                       data.size());
      }
    } else {
      return help(argv[0]);
    }
  }

  coded.resize(codedSize);

  if (outfileArg != nullptr) {
    bool const useStdout = outfileArg == std::string("-");
    if (useStdout) {
      f = stdout;
    } else {
      f = fopen(outfileArg, "wb");
    }
    fwrite(coded.data(), 1, coded.size(), f);
    if (!useStdout) {
      fclose(f);
    }
  }

  return 0;
}