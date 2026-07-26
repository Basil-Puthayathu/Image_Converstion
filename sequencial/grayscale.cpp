#include "grayscale.h"

cv::Mat convertToGrayscale(const cv::Mat& inputBGR) {
    // inputBGR is expected to be an 8-bit, 3-channel image (CV_8UC3).
    // Output is an 8-bit, 1-channel image (CV_8UC1) of the same dimensions.
    cv::Mat gray(inputBGR.rows, inputBGR.cols, CV_8UC1);

    for (int y = 0; y < inputBGR.rows; ++y) {
        // Row pointers avoid repeated .at<>() bounds-checking overhead.
        const cv::Vec3b* inRow = inputBGR.ptr<cv::Vec3b>(y);
        uchar* outRow = gray.ptr<uchar>(y);

        for (int x = 0; x < inputBGR.cols; ++x) {
            const cv::Vec3b& pixel = inRow[x];
            // OpenCV stores channels as B, G, R (in that order).
            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];

            double luminosity = 0.299 * R + 0.587 * G + 0.114 * B;

            // Clamp defensively in case of rounding drift, then cast.
            int value = static_cast<int>(luminosity + 0.5); // round to nearest
            if (value < 0) value = 0;
            if (value > 255) value = 255;

            outRow[x] = static_cast<uchar>(value);
        }
    }

    return gray;
}
