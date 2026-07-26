#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/opencv.hpp>
#include <array>

// Pipeline steps [2]-[5] from the design phase:
//   [2] Histogram Computation -> 256 intensity bins
//   [3] CDF Computation       -> running sum of histogram
//   [4] LUT Construction      -> normalized remap table
//   [5] LUT Application       -> final equalized image

// Step 2: count how many pixels fall into each of the 256 intensity bins.
std::array<long, 256> computeHistogram(const cv::Mat& gray);

// Step 3: running (cumulative) sum of the histogram.
std::array<long, 256> computeCDF(const std::array<long, 256>& histogram);

// Step 4: build the normalized remap table from the CDF.
// totalPixels is passed separately since it equals gray.rows * gray.cols.
std::array<uchar, 256> buildLUT(const std::array<long, 256>& cdf, long totalPixels);

// Step 5: apply the LUT to every pixel to produce the equalized image.
cv::Mat applyLUT(const cv::Mat& gray, const std::array<uchar, 256>& lut);

// Convenience wrapper that runs steps 2-5 in sequence and returns the
// final equalized image. Exposed separately so main_sequential.cpp can
// still time each individual step if finer-grained timing is wanted.
cv::Mat equalizeHistogram(const cv::Mat& gray);

#endif // HISTOGRAM_H
