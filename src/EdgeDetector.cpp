#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "EdgeDetector.h"

// Calculates edgemaps in H, S, and V space
cv::Mat EdgeDetector::findEdges(cv::Mat image)
{
	std::vector<cv::Mat> channels, edges;
	cv::split(image, channels);
	for (int i = 0; i < 3; i++)
	{
		edges[i] = gradient(channels[i]);
	}
	cv::Mat output;
	merge(edges, output);
	return output;
}

// Converts a 8-bit uint image to 0-1 double types
cv::Mat EdgeDetector::grayScaleToDouble(cv::Mat image)
{
	cv::Mat img = image;
	if (image.type() != CV_8U)
	{
		image.convertTo(img, CV_64F);
		img = img / 256.0;
	}
	return img;
}

// Filters an image using a custom convolution kernel
cv::Mat EdgeDetector::filterImage(cv::Mat image, cv::Mat filter)
{
	cv::Mat output;
	cv::filter2D(image, output, -1, filter);
	return output;
}

// Performs a convolution with a laplacian kernel
cv::Mat EdgeDetector::gradient(cv::Mat image)
{
	cv::Mat output;
	cv::filter2D(image, output, -1, cv::Mat(3, 3, CV_64F, laplacian));
	return output;
}

// Performs an convolution with an erosion kernel
cv::Mat EdgeDetector::erode(cv::Mat image)
{
	cv::Mat output;
	cv::erode(image, output, 0);
	return output;
}

// Thresholds the given image
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

// Thresholds the given image using HSV values
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

// Sums the pixel values of an image
cv::Vec3s EdgeDetector::sumPixels(cv::Mat image) // TODO unit test
{
	cv::Vec3b sum = cv::Vec3b({ 0, 0, 0 });
	for (int i = 0; i < image.size[0]; i++) for (int j = 0; j < image.size[1]; j++)
	{
		sum += image.at<cv::Vec3b>(i, j);
	}
	return sum;
}