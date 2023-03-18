#include <stdio.h>
#include <string>
#include <vector>

#include "ANSDecode.h"

static int help(char const *prog) {
  fprintf(stderr, "USAGE: %s RUNS INFILE [OUTFILE]\n\n", prog);
  fprintf(stderr,
          "Decompresses INFILE RUNS times and optionally writes to OUTFILE\n");
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    return help(argv[0]);
  }

  auto const runs = std::stoull(argv[1]);

  if (runs == 0) {
    return help(argv[0]);
  }

  FILE *f = fopen(argv[2], "rb");

  std::vector<uint8_t> data;
  data.resize(100 * 1024 * 1024);
  data.resize(fread(data.data(), 1, data.size(), f));
  fclose(f);

  std::vector<uint8_t> uncompressed;
  uncompressed.resize(100 * 1024 * 1024);

  size_t uncompressedSize;
  for (size_t i = 0; i < runs; ++i) {
    uncompressedSize = dietcpu::ansDecode(
        uncompressed.data(), uncompressed.size(), data.data(), data.size());
  }

  uncompressed.resize(uncompressedSize);

  if (argc == 4) {
    bool const useStdout = argv[3] == std::string("-");
    if (useStdout) {
        f = stdout;
    } else {
        f = fopen(argv[3], "wb");
    }
    fwrite(uncompressed.data(), 1, uncompressed.size(), f);
    if (!useStdout) {
        fclose(f);
    }
  }

  return 0;
}