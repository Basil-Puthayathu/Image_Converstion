#include "histogram_parallel.h"
#include <pthread.h>
#include <vector>

struct HistogramThreadArgs {
    const cv::Mat* gray;
    int startRow;
    int endRow;
    std::array<long, 256>* sharedHistogram; // the one final result, shared by all threads
    pthread_mutex_t* mutex;                 // guards access to sharedHistogram
};

void* histogramWorker(void* argPtr) {
    HistogramThreadArgs* args = static_cast<HistogramThreadArgs*>(argPtr);

    // ---- Phase 1: scan this thread's rows into a PRIVATE histogram ----
    // No lock needed here -- this array lives on this thread's own stack
    // and no other thread can see it.
    std::array<long, 256> localHistogram{};

    for (int y = args->startRow; y < args->endRow; ++y) {
        const uchar* row = args->gray->ptr<uchar>(y);
        for (int x = 0; x < args->gray->cols; ++x) {
            localHistogram[row[x]]++;
        }
    }

    // ---- Phase 2: merge into the shared histogram, protected by mutex ----
    pthread_mutex_lock(args->mutex);
    for (int i = 0; i < 256; ++i) {
        (*args->sharedHistogram)[i] += localHistogram[i];
    }
    pthread_mutex_unlock(args->mutex);

    return nullptr;
}

std::array<long, 256> computeHistogramParallel(const cv::Mat& gray, int numThreads) {
    std::array<long, 256> histogram{}; // shared final result, zero-initialized

    if (numThreads < 1) numThreads = 1;
    if (numThreads > gray.rows) numThreads = gray.rows;

    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    std::vector<pthread_t> threads(numThreads);
    std::vector<HistogramThreadArgs> args(numThreads);

    int rowsPerThread = gray.rows / numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? gray.rows : startRow + rowsPerThread;

        args[t] = { &gray, startRow, endRow, &histogram, &mutex };
        pthread_create(&threads[t], nullptr, histogramWorker, &args[t]);
    }

    for (int t = 0; t < numThreads; ++t) {
        pthread_join(threads[t], nullptr);
    }

    pthread_mutex_destroy(&mutex);

    return histogram;
}
