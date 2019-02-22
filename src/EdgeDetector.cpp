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
}

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

// Filters an image using a convolution
cv::Mat EdgeDetector::filterImage(cv::Mat image, cv::Mat filter)
{
	if (image.type() != CV_8U) printf("ERROR: gegeven image heeft weer de verkeerde datatypes: %i\n", image.type());
	cv::Mat img, output;
	image.convertTo(img, CV_64F);
	img = img / 256.0;
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

// Performs an erision convolution
cv::Mat EdgeDetector::erode(cv::Mat image)
{
	cv::Mat output;
	cv::erode(image, output, 0);
	return output;
}

// Thresholds the given image
cv::Mat EdgeDetector::threshold(cv::Mat image, double value)
{
	cv::Mat img;
	image.convertTo(img, CV_64F);
	img = img / 256.0;
	for (int i = 0; i < image.size[0]; i++) for (int j = 0; j < image.size[1]; j++)
	{
		if (img.at<double>(i, j) > value) ((double*)img.data)[j + i * img.size[1]] = 1.0;
		else ((double*)img.data)[j + i * img.size[1]] = 0.0;
	}
	return img;
}

// Thresholds the given image using HSV values
cv::Mat EdgeDetector::threshold(cv::Mat image, double H, double S, double V)
{
	cv::Mat img;
	cv::cvtColor(image, img, CV_BGR2HSV); // convert to HSV
	for (int i = 0; i < image.size[0]; i++) for (int j = 0; j < image.size[1]; j++)
	{
		// Do this for each channel (H, S, V)
		if (img.at<double>(i, j) > H) ((double*)img.data)[j + i * img.size[1]] = 1.0;
		else ((double*)img.data)[j + i * img.size[1]] = 0.0;
	}
	return img;
}

// Sums the pixel values of an image
float EdgeDetector::sumPixels(cv::Mat image)
{
	float sum = 0;
	for (int i = 0; i < image.size[0]; i++) for (int j = 0; j < image.size[1]; j++)
	{
		sum += image.at<double>(i, j);
	}
	return sum;
}