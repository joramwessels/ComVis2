
#include <random>

#include <opencv2/core/mat.hpp>

#include "EdgeDetector.h"

cv::Mat gradientDescent(cv::Mat frame, float learningRate)
{
	// input is edgemap of thresholded frame using estimate
	// target is edgemap of frame
	// find difference in HSV tensor
	// subtract difference from current estimate
	// if difference too small return estimate
	EdgeDetector ed;
	cv::Mat estimate; // TODO random init
	cv::Mat gradient, minGradient; // TODO set minimum gradient
	int iter, maxIter = 100;
	while (gradient < minGradient && iter < maxIter)
	{
		// Current estimate
		cv::Mat thresh = ed.thresholdHSV(frame, estimate.H, estimate.S, estimate.V);
		cv::Mat estimEdges = ed.findEdges(thresh);
		// Target
		cv::Mat targetEdges = ed.findEdges(frame);
		// Error
		cv::Mat error = targetEdges - estimEdges;
		gradient = ed.sumPixels(error);
		estimate += gradient * learningRate;
	}
	return estimate;
}