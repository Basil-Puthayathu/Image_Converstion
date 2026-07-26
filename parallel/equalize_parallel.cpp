#include "equalize_parallel.h"
#include <pthread.h>
#include <vector>

std::array<long, 256> computeCDF(const std::array<long, 256>& histogram) {
    std::array<long, 256> cdf{};
    long runningSum = 0;

    for (int i = 0; i < 256; ++i) {
        runningSum += histogram[i];
        cdf[i] = runningSum;
    }

    return cdf;
}

std::array<uchar, 256> buildLUT(const std::array<long, 256>& cdf, long totalPixels) {
    std::array<uchar, 256> lut{};

    long cdfMin = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] != 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    long denominator = totalPixels - cdfMin;
    if (denominator <= 0) {
        for (int i = 0; i < 256; ++i) lut[i] = static_cast<uchar>(i);
        return lut;
    }

    for (int i = 0; i < 256; ++i) {
        double normalized = static_cast<double>(cdf[i] - cdfMin) / denominator * 255.0;
        int value = static_cast<int>(normalized + 0.5);
        if (value < 0) value = 0;
        if (value > 255) value = 255;
        lut[i] = static_cast<uchar>(value);
    }

    return lut;
}

struct LutThreadArgs {
    const cv::Mat* gray;
    cv::Mat* output;
    const std::array<uchar, 256>* lut; // read-only, shared -- safe with no lock
    int startRow;
    int endRow;
};

void* lutWorker(void* argPtr) {
    LutThreadArgs* args = static_cast<LutThreadArgs*>(argPtr);

    for (int y = args->startRow; y < args->endRow; ++y) {
        const uchar* inRow = args->gray->ptr<uchar>(y);
        uchar* outRow = args->output->ptr<uchar>(y);
        for (int x = 0; x < args->gray->cols; ++x) {
            outRow[x] = (*args->lut)[inRow[x]];
        }
    }

    return nullptr;
}

cv::Mat applyLUTParallel(const cv::Mat& gray, const std::array<uchar, 256>& lut, int numThreads) {
    cv::Mat output(gray.rows, gray.cols, CV_8UC1);

    if (numThreads < 1) numThreads = 1;
    if (numThreads > gray.rows) numThreads = gray.rows;

    std::vector<pthread_t> threads(numThreads);
    std::vector<LutThreadArgs> args(numThreads);

    int rowsPerThread = gray.rows / numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? gray.rows : startRow + rowsPerThread;

        args[t] = { &gray, &output, &lut, startRow, endRow };
        pthread_create(&threads[t], nullptr, lutWorker, &args[t]);
    }

    for (int t = 0; t < numThreads; ++t) {
        pthread_join(threads[t], nullptr);
    }

    return output;
}

cv::Mat equalizeHistogramParallel(const cv::Mat& gray, const std::array<long, 256>& histogram, int numThreads) {
    long totalPixels = static_cast<long>(gray.rows) * gray.cols;

    std::array<long, 256> cdf = computeCDF(histogram);
    std::array<uchar, 256> lut = buildLUT(cdf, totalPixels);

    return applyLUTParallel(gray, lut, numThreads);
}
