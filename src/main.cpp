#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"

#include <opencv2/imgproc/imgproc.hpp>

#include "EdgeDetector.h"

using namespace nl_uu_science_gmt;

cv::Mat processForeground(cv::Mat foreground, cv::Mat background, uchar hThresh, uchar sThresh, uchar vThresh)
{
	cv::Mat result;

	cv::Mat foregroundHsv;
	cv::Mat backgroundHsv;
	cvtColor(foreground, foregroundHsv, CV_BGR2HSV);
	cvtColor(background, backgroundHsv, CV_BGR2HSV);

	std::vector<cv::Mat> foregroundChannels;
	std::vector<cv::Mat> backgroundChannels;
	split(foregroundHsv, foregroundChannels);
	split(backgroundHsv, backgroundChannels);

	// Background subtraction H
	cv::Mat tmp, fg, bg;
	absdiff(foregroundChannels[0], backgroundChannels[0], tmp);
	threshold(tmp, fg, hThresh, 255, CV_THRESH_BINARY);

	// Background subtraction S
	absdiff(foregroundChannels[1], backgroundChannels[1], tmp);
	threshold(tmp, bg, sThresh, 255, CV_THRESH_BINARY);
	bitwise_and(fg, bg, fg);

	// Background subtraction V
	absdiff(foregroundChannels[2], backgroundChannels[2], tmp);
	threshold(tmp, bg, vThresh, 255, CV_THRESH_BINARY);
	bitwise_or(fg, bg, fg);

	return fg;
}


// Returns the amount of different elements in two equally sized matrices
void matDiff(cv::Mat mat1, cv::Mat mat2) {
	cv::Mat result = mat1 - mat2;
	int diff = 0;

	for (int r = 0; r < result.rows; r++) {
		for (int c = 0; c < result.cols; c++) {
			cv::Vec3b px = result.at<cv::Vec3b>(r, c);
			if (!(px[0] == 0 && px[1] == 0 && px[2] == 0)) {
				diff++;
			}
		}
	}
}

int main(int argc, char** argv)
{
	//VoxelReconstruction::showKeys();
	//VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	//vr.run(argc, argv);

	cv::Mat background = cv::imread("data/cam1/background.png");
	cv::Mat foreground = cv::imread("data/cam1/foreground.png");

	cv::Mat processed = processForeground(foreground, background, 10, 10, 10);

	cv::imshow("Foreground", processed);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}