#ifndef EQUALIZE_PARALLEL_H
#define EQUALIZE_PARALLEL_H

#include <opencv2/opencv.hpp>
#include <array>

// Step [3] CDF Computation stays SEQUENTIAL: cdf[i] depends on cdf[i-1]
// (running sum), so there's a hard data dependency chain. Only 256
// elements total, so threading overhead would exceed the work saved.
std::array<long, 256> computeCDF(const std::array<long, 256>& histogram);

// Step [4] LUT Construction also stays SEQUENTIAL for the same reason.
std::array<uchar, 256> buildLUT(const std::array<long, 256>& cdf, long totalPixels);

// Step [5] LUT Application IS parallelized with pthreads. No mutex needed:
// the LUT is finished and read-only by this point, and every thread writes
// to a disjoint set of output rows.
cv::Mat applyLUTParallel(const cv::Mat& gray, const std::array<uchar, 256>& lut, int numThreads);

// Runs steps 3-5 in sequence (convenience wrapper).
cv::Mat equalizeHistogramParallel(const cv::Mat& gray, const std::array<long, 256>& histogram, int numThreads);

#endif // EQUALIZE_PARALLEL_H
