#pragma once

#include <opencv2/core/mat.hpp>

class EdgeDetector
{
protected:
	cv::Mat background;
	cv::Mat backgrEdges;
	double laplacian[9] = {
		0.0, -1.0, 0.0,
		-1.0, 4.0, -1.0,
		0.0, -1.0, 0.0
	};
public:
	EdgeDetector(cv::Mat background);
	~EdgeDetector() {};
	cv::Mat findEdges(cv::Mat image);
	cv::Mat findEdgesHSV(cv::Mat image);
	cv::Mat filterImage(cv::Mat image, cv::Mat filter);
	cv::Mat gradient(cv::Mat image);
	cv::Mat erode(cv::Mat image);
	cv::Vec3s sumPixels(cv::Mat image);
	cv::Mat threshold(cv::Mat image, unsigned char value);
	cv::Mat thresholdHSV(cv::Mat image, cv::Vec3b hsvThresh);
	cv::Mat getBackgrEdges() const { return backgrEdges; }
};