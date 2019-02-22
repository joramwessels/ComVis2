#pragma once

#include <opencv2/core/mat.hpp>

class EdgeDetector
{
protected:
	cv::Mat filter;
public:
	EdgeDetector() {};
	~EdgeDetector() {};
	cv::Mat filterImage(cv::Mat image);
	void setFilter(int size, double* values);
	float sumValues(cv::Mat image);
	cv::Mat threshold(cv::Mat image, double H, double S, double V);
};

