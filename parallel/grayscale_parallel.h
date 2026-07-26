#ifndef GRAYSCALE_PARALLEL_H
#define GRAYSCALE_PARALLEL_H

#include <opencv2/opencv.hpp>

// Parallel grayscale conversion using raw POSIX threads (pthreads).
//
// No mutex needed here: each thread is handed a disjoint block of rows and
// only ever writes to those rows in the output image. Different threads
// never touch the same memory, so there is nothing to protect.
cv::Mat convertToGrayscaleParallel(const cv::Mat& inputBGR, int numThreads);

#endif // GRAYSCALE_PARALLEL_H
