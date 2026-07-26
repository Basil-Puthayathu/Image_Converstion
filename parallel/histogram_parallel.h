#ifndef HISTOGRAM_PARALLEL_H
#define HISTOGRAM_PARALLEL_H

#include <opencv2/opencv.hpp>
#include <array>

// Parallel histogram computation using pthreads + an explicit mutex.
//
// The race condition: if every thread incremented a SHARED histogram
// array directly (histogram[pixelValue]++), two threads could read the
// same bucket's old value at the same time, both add 1, and both write
// back the same result -- one increment silently lost.
//
// The fix: each thread builds its own PRIVATE 256-bin histogram while
// scanning its rows (no sharing, no race, no lock needed for that part).
// Only when a thread is ready to merge its private counts into the
// shared final histogram does it lock a pthread_mutex_t, add its counts
// in, and unlock. The mutex ensures only one thread performs a merge at
// a time, so the shared histogram is never touched by two threads
// simultaneously.
std::array<long, 256> computeHistogramParallel(const cv::Mat& gray, int numThreads);

#endif // HISTOGRAM_PARALLEL_H
