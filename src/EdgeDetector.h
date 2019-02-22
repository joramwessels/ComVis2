#pragma once

#include <opencv2/core/mat.hpp>

class EdgeDetector
{
protected:
	double laplacian[9] = {
		0.0, -1.0, 0.0,
		-1.0, 4.0, -1.0,
		0.0, -1.0, 0.0
	};
public:
	EdgeDetector() {};
	~EdgeDetector() {};
	cv::Mat grayScaleToDouble(cv::Mat image);
	cv::Mat filterImage(cv::Mat image, cv::Mat filter);
	cv::Mat gradient(cv::Mat image);
	cv::Mat erode(cv::Mat image);
	float sumPixels(cv::Mat image);
	cv::Mat threshold(cv::Mat image, double value);
	cv::Mat threshold(cv::Mat image, double H, double S, double V);
};