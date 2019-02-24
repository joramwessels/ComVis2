#include <cstdlib>
#include <time.h>
#include <string>
#include <assert.h>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "EdgeDetector.h"
#include "utilities\General.h"
#include "GradientDescent.h"

inline bool belowMinimum(const cv::Vec3f a, const cv::Vec3f b) { return abs(a[0]) < abs(b[0]) && abs(a[1]) < abs(b[1]) && abs(a[2]) < abs(b[2]); }

/*
	Debug function that shows intermediate result

	@param image the image to display
	@param the interpretation of the image / the name of the window
*/
void showImage(cv::Mat image, const char* name="image debug")
{
	cv::Mat conv;
	image.convertTo(conv, CV_8U);
	cv::namedWindow(name);
	cv::imshow(name, conv);
	cv::waitKey(0);
	cv::destroyWindow(name);
}

/*
	Applies gradient descent to approximate the best HSV threshold values
	using the H, S, and V edgemaps as the target variables

	@param frame an image or video frame to determine the HSV values for (in HSV space)
	@param ed an EdgeDetector object pointer containing a background image
	@param learningRate the learning rate for the gradient descent
	@param maxIter the maximum amount of iterations alowed
	@param printEveryNIter the iteration step size at which to print the intermediate results
	@return the HSV threshold estimate
*/
cv::Vec3b gradientDescent(cv::Mat frame, EdgeDetector* ed, float learningRate, int maxIter, int printEveryNIter)
{
	printf("Starting gradient descent...\nlearning rate: %.3f\nmax iterations: %i\n", learningRate, maxIter);

	// Declarations
	cv::Mat hsvframe, targetEdges, diffWithBckgr, estimEdges, thresh, error;
	cv::Vec3d minGradient = cv::Vec3d({ .0005, .0005, .0005 }), gradient = minGradient;
	double gradientNorm = 1.0 / (double)(frame.size[0] * frame.size[1]);

	// Converting input
	cv::cvtColor(frame, hsvframe, CV_BGR2HSV);	// convert to HSV
	hsvframe.convertTo(hsvframe, CV_16S);		// convert to short

	// Precalculating target
	targetEdges = ed->findEdgesHSV(hsvframe); // is this result in CV_16S ?
	targetEdges = ed->computeError(targetEdges, ed->getBackgrEdges());
	//showImage(targetEdges, "target edges"); // DEBUG

	// Randomly initializing estimate
	cv::Vec3b estimate;
	cv::randu(estimate, cv::Scalar(10), cv::Scalar(245));
	cv::Vec3d estimateD = cv::Vec3d(estimate[0], estimate[1], estimate[2]);

	int iter = 0;
	while (!belowMinimum(gradient, minGradient) && iter < maxIter)
	{
		// Current estimate (thresholded difference with background)
		diffWithBckgr = hsvframe - ed->getBackground();
		thresh = ed->thresholdHSVSeparate(diffWithBckgr, estimate);
		estimEdges = ed->findEdgesHSV(thresh);

		// Get difference (blurred subtraction)
		error = ed->computeError(estimEdges, targetEdges);

		// update estimate
		gradient = ed->sumPixels(error) * gradientNorm;
		estimateD += gradient * learningRate;
		estimate[0] = (uchar)estimateD[0];
		estimate[1] = (uchar)estimateD[1];
		estimate[2] = (uchar)estimateD[2];

		if (iter % printEveryNIter == 0)
		{
			printf("iter: %i, error: (%.6f, %.6f, %.6f) ", iter, gradient[0], gradient[1], gradient[2]);
			printf("estim: (%i, %i, %i)\n", estimate[0], estimate[1], estimate[2]);
			showImage(diffWithBckgr, "diff with background"); // DEBUG
			showImage(ed->thresholdHSV(diffWithBckgr, estimate)); // DEBUG
			//showImage(thresh, "estimate threshold"); // DEBUG
			//showImage(estimEdges, "estimate edges"); // DEBUG
			//showImage(error, "error"); // DEBUG
		}
		iter++;
	}
	if (iter == maxIter) printf("iterated for %i loops. execution stopped.\n", maxIter);
	// DEBUG
	// Are the final edges even close to the target?
	cv::Mat estimU, targetU, errorU, diffU;
	estimEdges.convertTo(estimU, CV_8U);
	targetEdges.convertTo(targetU, CV_8U);
	error.convertTo(errorU, CV_8U);
	diffWithBckgr.convertTo(diffU, CV_8U);
	cv::namedWindow("Estimate");
	cv::namedWindow("Target");
	cv::namedWindow("Error");
	cv::imshow("Estimate", estimU);
	cv::imshow("Target", targetU);
	cv::imshow("Error", errorU);
	cv::waitKey(0);
	showImage(ed->thresholdHSV(diffWithBckgr, estimate), "result");
	// endDEBUG
	return estimate;
}

/*
	Loops through all videos and all frames and saves the HSV threshold estimates in the provided output file

	@param datafolder the path to the data folder
	@param outputfilename the output filename in which to write the HSV estimates per camera
	@param printEveryNFrames the iteration step size at which to print the intermediate results
*/
std::vector<cv::Vec3f> trainThresholdValues(const char* datafolder, const char* outputfilename, int printEveryNFrames)
{
	FILE* file = fopen(outputfilename, "w");
	int cam = 0;
	std::vector<cv::Vec3f> results(4);
	std::string path = datafolder + std::string("/");
	for (int i=0; i<4; i++)
	{
		std::string path = datafolder + std::string("/") + "cam" + std::to_string(i+1) + std::string("/");
		assert(nl_uu_science_gmt::General::fexists(path + "background.png"));
		cv::Mat background = cv::imread(path + "background.png");
		assert(nl_uu_science_gmt::General::fexists(path + "video.avi"));
		cv::VideoCapture cap(path + "video.avi");
		assert(cap.isOpened());

		EdgeDetector* ed = new EdgeDetector(background);
		int frameCount = 0;
		cv::Vec3d sum = cv::Vec3b({0, 0, 0});
		cv::Mat frame;
		int maxframes = 2; // DEBUG
		while (cap.read(frame))
		{
			if (frameCount >= maxframes) break; // DEBUG
			cv::Vec3b estimate = gradientDescent(frame, ed); // estimate HSV threshold
			sum += estimate;
			if (frameCount % printEveryNFrames == 0) printf("frame %i\n", frameCount);
			frameCount++;
		}
		cv::Vec3f avg = sum / frameCount;
		showImage(ed->thresholdHSV(frame, avg));
		fprintf(file, "cam%i: (%i, %i, %i)\n", cam + 1, (int)avg[0], (int)avg[1], (int)avg[2]);
		printf("avg for cam %i is (%.2f, %.2f, %.2f)\n", cam + 1, avg[0], avg[1], avg[2]);
		results[cam] = avg;
		cam++;
	}
	printf("\nTraining done\n");
	printf("cam1: (%.2f, %.2f, %.2f)\n", results[0][0], results[0][1], results[0][2]);
	printf("cam2: (%.2f, %.2f, %.2f)\n", results[1][0], results[1][1], results[1][2]);
	printf("cam3: (%.2f, %.2f, %.2f)\n", results[2][0], results[2][1], results[2][2]);
	printf("cam4: (%.2f, %.2f, %.2f)\n", results[3][0], results[3][1], results[3][2]);
	fclose(file);
	return results;
}

/*
	Reads HSV threshold values from a file
	
	@param filename the path to the file
	@param output a a destination vector of length 4
	@return a boolean indicating success
*/
bool readThresholds(const char* filename, std::vector<cv::Vec3f>&output) // TODO unit test
{
	std::ifstream file(filename);
	if (!file.is_open()) return false;
	std::string line;
	for (int i = 0; i < 4; i++)
	{
		if (!std::getline(file, line)) return false;
		if (!line.compare(0, 6, "cam" + std::to_string(i + 1) + ": (")) return false;
		int j = 0;
		//output[i][0] = std::stof(line.substr(7));
		//output[i][1] = std::stof(line.substr(j + 2), &j);
		//output[i][2] = std::stof(line.substr(j + 2), &j);
	}
	file.close();
}