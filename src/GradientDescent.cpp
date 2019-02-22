
#include <random>
#include <time.h>

#include <opencv2/core/mat.hpp>

#include "EdgeDetector.h"

inline bool belowMinimum(const cv::Vec3f a, const cv::Vec3f b) { return a[0] < b[0] && a[1] < b[1] && a[2] < b[2]; }

cv::Vec3b gradientDescent(cv::Mat frame, float learningRate=0.1f, int maxIter=1000)
{
	printf("Starting gradient descent...\nlearning rate: %.3f\nmax iterations: %i\n", learningRate, maxIter);
	int pixelCount = frame.size[0] * frame.size[1];
	srand(time(NULL));
	EdgeDetector ed;
	cv::Vec3b estimate = cv::Vec3b({ rand() % 256, rand() % 256, rand() % 256 });
	cv::Vec3f gradient, minGradient = cv::Vec3f({ 3.0f, 3.0f, 3.0f });
	cv::Mat thresh, estimEdges, targetEdges;
	cv::Mat error = cv::Mat(frame.size[0], frame.size[1], CV_16S);
	int iter;
	while (belowMinimum(gradient, minGradient) && iter < maxIter)
	{
		// Current estimate
		cv::Mat thresh = ed.thresholdHSV(frame, estimate);
		cv::Mat estimEdges = ed.findEdges(thresh);

		// Get error
		cv::Mat targetEdges = ed.findEdges(frame);
		cv::Mat error = (targetEdges - estimEdges);

		// update estimate
		gradient = ed.sumPixels(error) / pixelCount;
		estimate += gradient * learningRate;

		if (iter % 10 == 0) printf("iter: %i, error: (%.3f, %.3f, %.3f)\n", iter, gradient[0], gradient[1], gradient[2]);
	}
	return estimate;
}

void trainThresholdValues(const char* outputfilename)
{
	//FILE file = fopen(outputfilename, 'w');
	//int cam = 0;
	//std::vector<cv::Vec3f> results;
	//loop over cameras
	//{
	//		VideoCapture cap(data_path + checker_vid_fname);
	//		int frameCount = 0;
	//		cv::Vec3d sum = cv::Vec3b({0, 0, 0});
	//		loop over frames
	//		{
	//			cv::Vec3b estimate = gradientDescent(frame);
	//			sum += estimate;
	//			if (frameCount % 10 == 0) printf("frame %i\n", frameCount);
	//			framecount++;
	//		}
	//		cv::Vec3f avg = sum / frameCount;
	//		file << "cam" << (cam + 1) << ": (" << avg[0] << ", " << avg[1] << ", " << avg[2] << ")\n";
	//		printf("avg for cam %i is (%.2f, %.2f, %.2f)\n", cam + 1, avg[0], avg[1], avg[2]);
	//		results[cam] = avg;
	//		cam++;
	//}
}