#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "EdgeDetector.h"

/*
	Initializes the EdgeDetector object with the background and its edgemap

	@param background the RGB background image
*/
EdgeDetector::EdgeDetector(cv::Mat background) : background(background)
{
	cv::cvtColor(background, background, CV_BGR2HSV);
	backgrEdges = findEdgesHSV(background);
};

/*
	Calculates the edgemap of a grayscale (bitmap) image and returns it in 3 identical channels

	@param image the single channel (bitmap) image
*/
cv::Mat EdgeDetector::findEdges(cv::Mat image)
{
	std::vector<cv::Mat> channels(3);
	cv::Mat output, edges = gradient(image);
	for (int i = 0; i < 3; i++) channels[i] = edges / 3;
	cv::merge(channels, output);
	return output;
}

/*
	Calculates the average of the edgemaps for each channel

	@param image the triple channel (HSV) image
*/
cv::Mat EdgeDetector::findEdgesHSV(cv::Mat image)
{
	std::vector<cv::Mat> channels, edges(3);
	cv::split(image, channels);
	for (int i = 0; i < 3; i++)
	{
		edges[i] = gradient(channels[i]);
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
cv::Mat EdgeDetector::filterImage(cv::Mat image, cv::Mat filter)
{
	cv::Mat output;
	cv::filter2D(image, output, -1, filter);
	return output;
}

/*
	Performs a convolution with a laplacian kernel

	@param image the single channel image
*/
cv::Mat EdgeDetector::gradient(cv::Mat image)
{
	cv::Mat output;
	cv::filter2D(image, output, -1, cv::Mat(3, 3, CV_64F, laplacian));
	return output;
}

/*
	Performs an convolution with an erosion kernel

	@param image the single channel image
*/
cv::Mat EdgeDetector::erode(cv::Mat image)
{
	cv::Mat output;
	cv::erode(image, output, 0);
	return output;
}

/*
	Thresholds the given image

	@param image the single channel image
	@param value the pixel threshold value
*/
cv::Mat EdgeDetector::threshold(cv::Mat image, unsigned char value)
{
	cv::Mat img;
	for (int i = 0; i < image.size[0]; i++) for (int j = 0; j < image.size[1]; j++)
	{
		if (img.at<unsigned char>(i, j) > value) ((unsigned char*)img.data)[j + i * img.size[1]] = 255;
		else ((unsigned char*)img.data)[j + i * img.size[1]] = 0;
	}
	return img;
}

/*
	Thresholds the given image using HSV values

	@param foreground the tripe channel image including foreground
	@param hsvThresh the HSV threshold values
*/
cv::Mat EdgeDetector::thresholdHSV(cv::Mat foreground, cv::Vec3b hsvThresh)
{
	cv::Mat result;

	cv::Mat foregroundHsv;
	cv::Mat backgroundHsv;
	cv::cvtColor(foreground, foregroundHsv, CV_BGR2HSV);
	cv::cvtColor(background, backgroundHsv, CV_BGR2HSV);

	std::vector<cv::Mat> foregroundChannels;
	std::vector<cv::Mat> backgroundChannels;
	cv::split(foregroundHsv, foregroundChannels);
	cv::split(backgroundHsv, backgroundChannels);

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
	Sums the pixel values of an image

	@param image the triple channel image (containing cv::Vec3s types)
*/
cv::Vec3s EdgeDetector::sumPixels(cv::Mat image)
{
	double pixelCount = 1 / image.size[0] * image.size[1];
	cv::Vec3d sum = cv::Vec3d({ 0, 0, 0 });
	for (int i = 0; i < image.size[0]; i++) for (int j = 0; j < image.size[1]; j++)
	{
		//sum += image.at<cv::Vec3s>(i, j) * pixelCount;
		sum[0] += (double)(image.at<cv::Vec3s>(i, j)[0]) * pixelCount;
		sum[1] += (double)(image.at<cv::Vec3s>(i, j)[1]) * pixelCount;
		sum[2] += (double)(image.at<cv::Vec3s>(i, j)[2]) * pixelCount;
	}
	return sum;
}