#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "EdgeDetector.h"

/*
	Initializes the EdgeDetector object with the background and its edgemap

	@param background the RGB background image
*/
EdgeDetector::EdgeDetector(cv::Mat background) : m_background(background)
{
	cv::cvtColor(m_background, m_background, CV_BGR2HSV);
	m_background.convertTo(m_background, CV_16S);
	m_backgrEdges = findEdgesHSV(m_background);
	m_backgrEdges.convertTo(m_backgrEdges, CV_16S);
};

/*
	Calculates the edgemap of a single channel image
	
	@param image the single channel image
*/
cv::Mat EdgeDetector::findEdges(cv::Mat image)
{
	cv::Mat output = image;
	erode(output);
	blur(output);
	gradient(output);
	dilate(output);
	return output;
}

/*
	Calculates triple channel edgemap from single channel input

	@param image the single channel (bitmap) image
*/
cv::Mat EdgeDetector::findEdgesSingle(cv::Mat image)
{
	std::vector<cv::Mat> channels(3);
	cv::Mat output, edges = findEdges(image);
	for (int i = 0; i < 3; i++) channels[i] = edges / 3;
	cv::merge(channels, output);
	return output;
}

/*
	Calculates the edgemap of each HSV channel separately

	@param image the triple channel (HSV) image
*/
cv::Mat EdgeDetector::findEdgesHSV(cv::Mat image)
{
	std::vector<cv::Mat> channels, edges(3);
	cv::split(image, channels);
	for (int i = 0; i < 3; i++)
	{
		edges[i] = findEdges(channels[i]);
	}
	cv::Mat output;
	cv::merge(edges, output);
	return output;
}

/*
	Filters an image using a custom convolution kernel

	@param image the single channel image
	@param the convolution kernel
*/
void EdgeDetector::filterImage(cv::Mat image, cv::Mat filter)
{
	cv::filter2D(image, image, -1, filter);
}

/*
	Performs a convolution with a laplacian kernel

	@param image the single channel image
*/
void EdgeDetector::gradient(cv::Mat image)
{
	cv::filter2D(image, image, -1, cv::Mat(3, 3, CV_64F, laplacian));
}

/*
	Performs a convolution with an erosion kernel

	@param image the single channel image
*/
void EdgeDetector::erode(cv::Mat image)
{
	cv::erode(image, image, 0);
}

/*
	Performs a convolution with a dilation kernel
	
	@param image the single channel image
*/
void EdgeDetector::dilate(cv::Mat image)
{
	cv::dilate(image, image, 0);
}

/*
	Performs a convolution with a Gaussian kernel
	
	@param image the single channel image
*/
void EdgeDetector::blur(cv::Mat image)
{
	cv::blur(image, image, cv::Size(3, 3));
}

/*
	Thresholds a single channel image
	If the element reaches the threshold, it retains its previous value

	@param image the single channel image (with CV_16S types)
	@param value the pixel threshold value
*/
void EdgeDetector::threshold(cv::Mat image, unsigned char value)
{
	int rows = image.size[0], cols = image.size[1];
	for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++)
	{
		if (abs(image.at<short>(r, c)) < value)
			((short*)image.data)[c + r * cols] = 0;
	}
}

/*
	Thresholds the given image using HSV values

	@param foreground the tripe channel image including foreground
	@param hsvThresh the HSV threshold values
*/
cv::Mat EdgeDetector::thresholdHSV(cv::Mat foreground, cv::Vec3b hsvThresh)
{
	cv::Mat result;

	cv::Mat foregroundU;
	cv::Mat backgroundU;
	foreground.convertTo(foregroundU, CV_8U);
	m_background.convertTo(backgroundU, CV_8U);

	std::vector<cv::Mat> foregroundChannels;
	std::vector<cv::Mat> backgroundChannels;
	cv::split(foregroundU, foregroundChannels);
	cv::split(backgroundU, backgroundChannels);

	// Background subtraction H
	cv::Mat tmp, fg, bg;
	cv::absdiff(foregroundChannels[0], backgroundChannels[0], tmp);
	cv::threshold(tmp, fg, hsvThresh[0], 255, CV_THRESH_BINARY);

	// Background subtraction S
	cv::absdiff(foregroundChannels[1], backgroundChannels[1], tmp);
	cv::threshold(tmp, bg, hsvThresh[1], 255, CV_THRESH_BINARY);
	cv::bitwise_and(fg, bg, fg);

	// Background subtraction V
	cv::absdiff(foregroundChannels[2], backgroundChannels[2], tmp);
	cv::threshold(tmp, bg, hsvThresh[2], 255, CV_THRESH_BINARY);
	cv::bitwise_or(fg, bg, fg);

	return fg;
}

/*
	Thresholds each channel of the image only with its respective H, S, or V threshold
	and returns a triple channel bitmap
	
	@param foreground the tripe channel image including foreground
	@param hsvThresh the HSV threshold values
*/
cv::Mat EdgeDetector::thresholdHSVSeparate(const cv::Mat image, cv::Vec3b hsvThresh)
{
	std::vector<cv::Mat> channels;
	cv::split(image, channels);
	for (int i = 0; i < 3; i++) threshold(channels[i], hsvThresh[i]);
	cv::Mat output;
	cv::merge(channels, output);
	return output;
}

/*
	Subtracts the blurred versions of the edgemaps to get the difference
	
	@param image1
	@param image2
*/
cv::Mat EdgeDetector::computeError(cv::Mat image1, cv::Mat image2)
{
	cv::Mat blur1, blur2;
	std::vector<cv::Mat> channels;

	cv::split(image1, channels);
	for (int i = 0; i < 3; i++) blur(channels[i]);
	cv::merge(channels, blur1);

	cv::split(image2, channels);
	for (int i = 0; i < 3; i++) blur(channels[i]);
	cv::merge(channels, blur2);

	return image1 - image2;
}

/*
	Sums the pixel values of an image

	@param image the triple channel image (containing cv::Vec3s types)
*/
cv::Vec3d EdgeDetector::sumPixels(cv::Mat image)
{
	cv::Vec3d sum = cv::Vec3d(0, 0, 0);
	for (int r = 0; r < image.rows; r++) for (int c = 0; c < image.cols; c++) {
		cv::Vec3s px = image.at<cv::Vec3s>(r, c);
		sum[0] += (double) px[0];
		sum[1] += (double) px[1];
		sum[2] += (double) px[2];
	}
	return sum;
}