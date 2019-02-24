#pragma once

#include <opencv2/core/mat.hpp>
#include "EdgeDetector.h"

cv::Vec3b gradientDescent(cv::Mat frame, EdgeDetector* ed, float learningRate = 100.0f, int maxIter = 1000, int printEveryNIter = 100);
std::vector<cv::Vec3f> trainThresholdValues(const char* datafolder, const char* outputfilename, int printEveryNFrames=1);
bool readThresholds(const char* filename, std::vector<cv::Vec3f>&output);