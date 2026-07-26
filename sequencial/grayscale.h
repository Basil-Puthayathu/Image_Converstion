#ifndef GRAYSCALE_H
#define GRAYSCALE_H

#include <opencv2/opencv.hpp>

// Converts a BGR (OpenCV's default channel order) image to a single-channel
// grayscale image using the luminosity formula:
//     gray = 0.299*R + 0.587*G + 0.114*B
//
// Implemented manually (pixel-by-pixel) rather than via cv::cvtColor so the
// cost of this stage is real and comparable against the parallel version.
cv::Mat convertToGrayscale(const cv::Mat& inputBGR);

#endif // GRAYSCALE_H
