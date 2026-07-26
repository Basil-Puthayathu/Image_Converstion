#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "grayscale.h"
#include "histogram.h"

// Small helper so every stage is timed the same way.
using Clock = std::chrono::high_resolution_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    // Usage: ./main_sequential <input_image> [output_image]
    std::string inputPath = (argc > 1) ? argv[1] : "input.jpg";
    std::string outputPath = (argc > 2) ? argv[2] : "output_equalized.png";

    std::cout << "===== Sequential Baseline: Grayscale + Histogram Equalization =====\n";
    std::cout << "Input : " << inputPath << "\n";
    std::cout << "Output: " << outputPath << "\n\n";

    // ---- Stage 1: Load image ----
    auto t0 = Clock::now();
    cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_COLOR);
    auto t1 = Clock::now();

    if (inputImage.empty()) {
        std::cerr << "Error: could not load image at '" << inputPath << "'\n";
        return 1;
    }
    double loadTime = elapsedMs(t0, t1);

    // ---- Stage 2: Grayscale conversion ----
    auto t2 = Clock::now();
    cv::Mat grayImage = convertToGrayscale(inputImage);
    auto t3 = Clock::now();
    double grayscaleTime = elapsedMs(t2, t3);

    // ---- Stage 3: Histogram equalization (steps 2-5 of the pipeline) ----
    auto t4 = Clock::now();
    cv::Mat equalizedImage = equalizeHistogram(grayImage);
    auto t5 = Clock::now();
    double equalizeTime = elapsedMs(t4, t5);

    // ---- Stage 4: Save output image ----
    auto t6 = Clock::now();
    bool saved = cv::imwrite(outputPath, equalizedImage);
    auto t7 = Clock::now();
    double saveTime = elapsedMs(t6, t7);

    if (!saved) {
        std::cerr << "Error: could not write output image to '" << outputPath << "'\n";
        return 1;
    }

    double totalTime = loadTime + grayscaleTime + equalizeTime + saveTime;

    // ---- Report timings ----
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Stage                          Time (ms)\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "1. Image loading             : " << loadTime      << " ms\n";
    std::cout << "2. Grayscale conversion      : " << grayscaleTime << " ms\n";
    std::cout << "3. Histogram equalization    : " << equalizeTime  << " ms\n";
    std::cout << "4. Image saving              : " << saveTime      << " ms\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "TOTAL (measured stages)      : " << totalTime     << " ms\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Image size: " << inputImage.cols << " x " << inputImage.rows
              << " (" << (inputImage.cols * inputImage.rows) << " pixels)\n";

    return 0;
}
