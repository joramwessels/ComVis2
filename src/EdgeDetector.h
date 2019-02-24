#pragma once

#include <opencv2/core/mat.hpp>

class EdgeDetector
{
protected:
	cv::Mat m_background;
	cv::Mat m_backgrEdges;
	double laplacian[9] = {
		0.0, -1.0, 0.0,
		-1.0, 4.0, -1.0,
		0.0, -1.0, 0.0
	};
public:
	EdgeDetector(cv::Mat background);
	~EdgeDetector() {};
	cv::Mat findEdges(cv::Mat image);
	cv::Mat findEdgesSingle(cv::Mat image);
	cv::Mat findEdgesHSV(cv::Mat image);
	void filterImage(cv::Mat image, cv::Mat filter);
	void gradient(cv::Mat image);
	void erode(cv::Mat image);
	void dilate(cv::Mat image);
	void blur(cv::Mat image);
	void threshold(cv::Mat image, unsigned char value);
	cv::Mat thresholdHSV(cv::Mat image, cv::Vec3b hsvThresh);
	cv::Mat thresholdHSVSeparate(cv::Mat foreground, cv::Vec3b hsvThresh);
	cv::Mat computeError(cv::Mat image1, cv::Mat image2);
	cv::Vec3d sumPixels(cv::Mat image);
	cv::Mat getBackground() const { return m_background; }
	cv::Mat getBackgrEdges() const { return m_backgrEdges; }
};