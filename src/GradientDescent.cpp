#include <cstdlib>
#include <time.h>
#include <string>
#include <assert.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "EdgeDetector.h"
#include "utilities\General.h"
#include "GradientDescent.h"

inline bool belowMinimum(const cv::Vec3f a, const cv::Vec3f b) { return a[0] < b[0] && a[1] < b[1] && a[2] < b[2]; }

/*
	Debug function that shows intermediate result

	@param image the image to display
	@param the interpretation of the image / the name of the window
*/
void showImage(cv::Mat image, const char* name="image debug")
{
	cv::namedWindow(name);
	cv::imshow(name, image);
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
	float pixelCount = frame.size[0] * frame.size[1];
	cv::Vec3b estimate;
	cv::randu(estimate, cv::Scalar(0), cv::Scalar(255));
	cv::Vec3d minGradient = cv::Vec3d({ 1.0, 1.0, 1.0 }), gradient = minGradient;
	cv::Mat thresh, estimEdges, targetEdges;
	cv::Mat error = cv::Mat(frame.size[0], frame.size[1], CV_16S);
	int iter = 0;
	while (!belowMinimum(gradient, minGradient) && iter < maxIter)
	{
		// Current estimate
		thresh = ed->thresholdHSV(frame, estimate);
		showImage(thresh, "thresholded estimate"); // DEBUG
		estimEdges = ed->findEdges(thresh);
		showImage(estimEdges, "estimate edgemap"); // DEBUG

		// Get difference
		targetEdges = ed->findEdgesHSV(frame);
		targetEdges -= ed->getBackgrEdges();
		showImage(targetEdges, "target edgemap"); // DEBUG
		error = (targetEdges - estimEdges);
		showImage(error, "edgemap difference"); // DEBUG

		// update estimate
		gradient = ed->sumPixels(error);
		estimate += gradient * learningRate;

		if (iter % printEveryNIter == 0) printf("iter: %i, error: (%.3f, %.3f, %.3f)\n", iter, gradient[0], gradient[1], gradient[2]);
		iter++;
	}
	if (iter == maxIter) printf("iterated for %i loops. execution stopped.\n", maxIter);
	return estimate;
}

/*
	Loops through all videos and all frames and saves the HSV threshold estimates in the provided output file

	@param datafolder the path to the data folder
	@param outputfilename the output filename in which to write the HSV estimates per camera
	@param printEveryNFrames the iteration step size at which to print the intermediate results
*/
void trainThresholdValues(const char* datafolder, const char* outputfilename, int printEveryNFrames)
{
	FILE* file = fopen(outputfilename, "w");
	int cam = 0;
	std::vector<cv::Vec3f> results;
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
		cv::Mat frame, hsvframe;
		int maxframes = 2; // DEBUG
		while (cap.read(frame))
		{
			if (frameCount >= maxframes) break; // DEBUG
			cv::cvtColor(frame, hsvframe, CV_BGR2HSV);		 // convert to HSV
			cv::Vec3b estimate = gradientDescent(hsvframe, ed); // estimate HSV threshold
			sum += estimate;
			if (frameCount % printEveryNFrames == 0) printf("frame %i\n", frameCount);
			frameCount++;
		}
		cv::Vec3f avg = sum / frameCount;
		fprintf(file, "cam%i: (%.2f, %.2f, %.2f)\n", cam + 1, avg[0], avg[1], avg[2]);
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
}