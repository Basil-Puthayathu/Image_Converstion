#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread> // only used for std::thread::hardware_concurrency()

#include "grayscale_parallel.h"
#include "histogram_parallel.h"
#include "equalize_parallel.h"

using Clock = std::chrono::high_resolution_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    // Usage: ./main_parallel <input_image> [output_image] [num_threads]
    std::string inputPath = (argc > 1) ? argv[1] : "input.jpg";
    std::string outputPath = (argc > 2) ? argv[2] : "output_equalized_parallel.png";

    int numThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (numThreads <= 0) numThreads = 4;

    if (argc > 3) {
        int requested = std::atoi(argv[3]);
        if (requested > 0) numThreads = requested;
    }

    std::cout << "===== Parallel Pipeline: Grayscale + Histogram Equalization (pthreads) =====\n";
    std::cout << "Input        : " << inputPath << "\n";
    std::cout << "Output       : " << outputPath << "\n";
    std::cout << "Threads used : " << numThreads << "\n\n";

    // ---- Stage 1: Load image (disk I/O, not parallelized) ----
    auto t0 = Clock::now();
    cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_COLOR);
    auto t1 = Clock::now();

    if (inputImage.empty()) {
        std::cerr << "Error: could not load image at '" << inputPath << "'\n";
        return 1;
    }
    double loadTime = elapsedMs(t0, t1);

    // ---- Stage 2: Grayscale conversion (pthreads) ----
    auto t2 = Clock::now();
    cv::Mat grayImage = convertToGrayscaleParallel(inputImage, numThreads);
    auto t3 = Clock::now();
    double grayscaleTime = elapsedMs(t2, t3);

    // ---- Stage 3: Histogram equalization ----
    // Broken into its 4 sub-steps with their own chrono timers: histogram
    // computation (parallel) + CDF computation (sequential) + LUT
    // construction (sequential) + LUT application (parallel).
    long totalPixels = static_cast<long>(grayImage.rows) * grayImage.cols;

    auto histStart = Clock::now();
    std::array<long, 256> histogram = computeHistogramParallel(grayImage, numThreads);
    auto histEnd = Clock::now();
    double histogramTime = elapsedMs(histStart, histEnd);

    auto cdfStart = Clock::now();
    std::array<long, 256> cdf = computeCDF(histogram);
    auto cdfEnd = Clock::now();
    double cdfTime = elapsedMs(cdfStart, cdfEnd);

    auto lutBuildStart = Clock::now();
    std::array<uchar, 256> lut = buildLUT(cdf, totalPixels);
    auto lutBuildEnd = Clock::now();
    double lutBuildTime = elapsedMs(lutBuildStart, lutBuildEnd);

    auto lutApplyStart = Clock::now();
    cv::Mat equalizedImage = applyLUTParallel(grayImage, lut, numThreads);
    auto lutApplyEnd = Clock::now();
    double lutApplyTime = elapsedMs(lutApplyStart, lutApplyEnd);

    double equalizeTime = histogramTime + cdfTime + lutBuildTime + lutApplyTime;

    // ---- Stage 4: Save output image (disk I/O, not parallelized) ----
    auto t6 = Clock::now();
    bool saved = cv::imwrite(outputPath, equalizedImage);
    auto t7 = Clock::now();
    double saveTime = elapsedMs(t6, t7);

    if (!saved) {
        std::cerr << "Error: could not write output image to '" << outputPath << "'\n";
        return 1;
    }

    double totalTime = loadTime + grayscaleTime + equalizeTime + saveTime;
    double computeOnlyTime = grayscaleTime + equalizeTime;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Stage                          Time (ms)\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "1. Image loading             : " << loadTime      << " ms\n";
    std::cout << "2. Grayscale conversion      : " << grayscaleTime << " ms\n";
    std::cout << "3. Histogram equalization    : " << equalizeTime  << " ms\n";
    std::cout << "4. Image saving              : " << saveTime      << " ms\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Stage 3 breakdown:\n";
    std::cout << "  3a. Histogram computation   (parallel)   : " << histogramTime << " ms\n";
    std::cout << "  3b. CDF computation          (sequential) : " << cdfTime       << " ms\n";
    std::cout << "  3c. LUT construction         (sequential) : " << lutBuildTime  << " ms\n";
    std::cout << "  3d. LUT application          (parallel)   : " << lutApplyTime  << " ms\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "TOTAL (measured stages)      : " << totalTime     << " ms\n";
    std::cout << "COMPUTE ONLY (stage 2+3)     : " << computeOnlyTime << " ms\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Image size: " << inputImage.cols << " x " << inputImage.rows
              << " (" << (inputImage.cols * inputImage.rows) << " pixels)\n";

    return 0;
}
