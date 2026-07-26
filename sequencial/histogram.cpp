#include "histogram.h"

std::array<long, 256> computeHistogram(const cv::Mat& gray) {
    std::array<long, 256> histogram{}; // zero-initialized

    for (int y = 0; y < gray.rows; ++y) {
        const uchar* row = gray.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            histogram[row[x]]++;
        }
    }

    return histogram;
}

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

    // Find cdf_min: the smallest non-zero cumulative value.
    // This is the standard normalization anchor for histogram equalization.
    long cdfMin = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] != 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    // Guard against a degenerate image (e.g. every pixel the same value),
    // which would otherwise divide by zero.
    long denominator = totalPixels - cdfMin;
    if (denominator <= 0) {
        for (int i = 0; i < 256; ++i) lut[i] = static_cast<uchar>(i);
        return lut;
    }

    for (int i = 0; i < 256; ++i) {
        double normalized = static_cast<double>(cdf[i] - cdfMin) / denominator * 255.0;
        int value = static_cast<int>(normalized + 0.5); // round to nearest
        if (value < 0) value = 0;
        if (value > 255) value = 255;
        lut[i] = static_cast<uchar>(value);
    }

    return lut;
}

cv::Mat applyLUT(const cv::Mat& gray, const std::array<uchar, 256>& lut) {
    cv::Mat output(gray.rows, gray.cols, CV_8UC1);

    for (int y = 0; y < gray.rows; ++y) {
        const uchar* inRow = gray.ptr<uchar>(y);
        uchar* outRow = output.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            outRow[x] = lut[inRow[x]];
        }
    }

    return output;
}

cv::Mat equalizeHistogram(const cv::Mat& gray) {
    long totalPixels = static_cast<long>(gray.rows) * gray.cols;

    std::array<long, 256> histogram = computeHistogram(gray);
    std::array<long, 256> cdf = computeCDF(histogram);
    std::array<uchar, 256> lut = buildLUT(cdf, totalPixels);

    return applyLUT(gray, lut);
}
