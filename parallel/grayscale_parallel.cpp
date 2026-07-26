#include "grayscale_parallel.h"
#include <pthread.h>
#include <vector>

// Everything a worker thread needs to know to do its share of the work.
// pthread_create only accepts a single void* argument, so we bundle
// everything the thread needs into one struct and pass its address.
struct GrayscaleThreadArgs {
    const cv::Mat* input;
    cv::Mat* output;
    int startRow; // inclusive
    int endRow;   // exclusive
};

// The function each thread actually runs. pthreads requires this exact
// signature: void* (*)(void*).
void* grayscaleWorker(void* argPtr) {
    GrayscaleThreadArgs* args = static_cast<GrayscaleThreadArgs*>(argPtr);

    for (int y = args->startRow; y < args->endRow; ++y) {
        const cv::Vec3b* inRow = args->input->ptr<cv::Vec3b>(y);
        uchar* outRow = args->output->ptr<uchar>(y);

        for (int x = 0; x < args->input->cols; ++x) {
            const cv::Vec3b& pixel = inRow[x];
            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];

            double luminosity = 0.299 * R + 0.587 * G + 0.114 * B;

            int value = static_cast<int>(luminosity + 0.5);
            if (value < 0) value = 0;
            if (value > 255) value = 255;

            outRow[x] = static_cast<uchar>(value);
        }
    }

    return nullptr;
}

cv::Mat convertToGrayscaleParallel(const cv::Mat& inputBGR, int numThreads) {
    cv::Mat gray(inputBGR.rows, inputBGR.cols, CV_8UC1);

    if (numThreads < 1) numThreads = 1;
    if (numThreads > inputBGR.rows) numThreads = inputBGR.rows;

    std::vector<pthread_t> threads(numThreads);
    std::vector<GrayscaleThreadArgs> args(numThreads);

    int rowsPerThread = inputBGR.rows / numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? inputBGR.rows : startRow + rowsPerThread;

        args[t] = { &inputBGR, &gray, startRow, endRow };
        pthread_create(&threads[t], nullptr, grayscaleWorker, &args[t]);
    }

    for (int t = 0; t < numThreads; ++t) {
        pthread_join(threads[t], nullptr);
    }

    return gray;
}
