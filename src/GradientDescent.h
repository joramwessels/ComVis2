#pragma once

#include <opencv2/core/mat.hpp>
#include "EdgeDetector.h"

cv::Vec3b gradientDescent(cv::Mat frame, EdgeDetector* ed, float learningRate = .1f, int maxIter = 1000, int printEveryNIter = 10);
void trainThresholdValues(const char* datafolder, const char* outputfilename, int printEveryNFrames=10);