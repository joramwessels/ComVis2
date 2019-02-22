
#include <random>

#include <opencv2/core/mat.hpp>

#include "EdgeDetector.h"

cv::Mat gradientDescent(cv::Mat frame, float learningRate=1.0f)
{
	EdgeDetector ed;
	cv::Mat estimate; // TODO random init
	cv::Mat gradient, minGradient; // TODO set minimum gradient
	int iter, maxIter = 100;
	while (gradient < minGradient && iter < maxIter) // TODO convergence criteria
	{
		// Current estimate
		cv::Mat thresh = ed.thresholdHSV(frame, estimate);
		cv::Mat estimEdges = ed.findEdges(thresh);

		// Get error
		cv::Mat targetEdges = ed.findEdges(frame);
		cv::Mat error = targetEdges - estimEdges;

		// update estimate
		gradient = ed.sumPixels(error);
		estimate += gradient * learningRate;
	}
	return estimate;
}