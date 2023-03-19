# DietCPU

Experimental proof of concept to show that the DietGPU format can be efficiently encoded & decoded on a CPU.
I've written a direct translation of the DietGPU code, and haven't spent much time optimizing, because all I want to show is that we can be significantly faster than scalar ANS with this format.
There are probably some followups that we could do that would involve breaking the DietGPU format, which could potentially get us some gains, but might come at the cost of GPU performance.

## Benchmarks

I've done all benchmarks on an Intel i9-9900K with Turbo disabled. I've compiled with clang-14.0.6.
I tested on files generated with a `std::exponential_distribution<float>(lambda)` for the listed `lambda`. I encoded & decoded with the given `probBits`.

| lambda | probBits | input size | encoded size | encode speed | decode speed | FSE encoded size | FSE encode speed | FSE decode speed |
|--------|----------|------------|--------------|--------------|--------------|------------------|------------------|------------------|
|      1 |        9 |    512 KiB |       412760 |    1269 MB/s |    1828 MB/s |           392417 |         530 MB/s |         571 MB/s |
|      1 |       10 |    512 KiB |       407276 |    1268 MB/s |    1827 MB/s |           391482 |         529 MB/s |         571 MB/s |
|      1 |       11 |    512 KiB |       404678 |    1289 MB/s |    1822 MB/s |           390983 |         529 MB/s |         570 MB/s |
|     10 |        9 |    512 KiB |       445070 |    1280 MB/s |    1825 MB/s |           422249 |         529 MB/s |         570 MB/s |
|     10 |       10 |    512 KiB |       426872 |    1280 MB/s |    1824 MB/s |           408841 |         530 MB/s |         570 MB/s |
|     10 |       11 |    512 KiB |       419568 |    1277 MB/s |    1816 MB/s |           404128 |         530 MB/s |         569 MB/s |
|    100 |        9 |    512 KiB |       201080 |    1323 MB/s |    1850 MB/s |           186092 |         528 MB/s |         578 MB/s |
|    100 |       10 |    512 KiB |       199262 |    1325 MB/s |    1851 MB/s |           184873 |         528 MB/s |         577 MB/s |
|    100 |       11 |    512 KiB |       198452 |    1324 MB/s |    1848 MB/s |           184368 |         528 MB/s |         577 MB/s |
|   1000 |        9 |    512 KiB |        23030 |    1545 MB/s |    1879 MB/s |             9712 |         528 MB/s |         574 MB/s |
|   1000 |       10 |    512 KiB |        23036 |    1537 MB/s |    1877 MB/s |             9603 |         528 MB/s |         574 MB/s |
|   1000 |       11 |    512 KiB |        23040 |    1514 MB/s |    1877 MB/s |             9566 |         527 MB/s |         572 MB/s |

Note: The encoding speedup at lambda=1000 happens because I have a specialization that can handle max symbol value < 16 without doing any gathers.