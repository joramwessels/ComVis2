
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "EdgeDetector.h"

// Sets the convolution kernel
void EdgeDetector::setFilter(int size, double* values)
{
	filter = cv::Mat(size, size, CV_64F, values);
}

// Filters an image using a convolution
cv::Mat EdgeDetector::filterImage(cv::Mat image)
{
	if (image.type() != CV_8U) printf("ERROR: gegeven image heeft weer de verkeerde datatypes: %i\n", image.type());
	cv::Mat img, output;
	image.convertTo(img, CV_64F);
	img = img / 256.0;
	cv::filter2D(image, output, -1, filter);
	return output;
}

// Shows bitmap image with the given HSV thresholding
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
float EdgeDetector::sumValues(cv::Mat image)
{
	float sum = 0;
	for (int i = 0; i < image.size[0]; i++) for (int j = 0; j < image.size[1]; j++)
	{
		sum += image.at<double>(i, j);
	}
	return sum;
}