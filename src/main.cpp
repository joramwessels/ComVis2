#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include <climits>

#include <opencv2/imgproc/imgproc.hpp>

#include "GradientDescent.h"

using namespace nl_uu_science_gmt;

// Returns the average frame (average color per pixel) of a video as a BGR matrix
cv::Mat averageFrame(const std::string &path) {
	cv::VideoCapture cap(path);

	cv::Mat result(cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH), CV_8UC3);
	std::vector<cv::Mat> frames;
	const int frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
	
	for (int i = 0; i < frameCount; i++) {
		cv::Mat frame;
		cap >> frame;

		for (int i = 0; i < frameCount; i++) {
			frames.push_back(frame);
		}
	}

	for (int r = 0; r < result.rows; r++) {
		for (int c = 0; c < result.cols; c++) {
			printf("\nPixel %d/%d", r * result.cols + c, result.rows * result.cols);
			int totalB = 0;
			int totalG = 0;
			int totalR = 0;

			for (int i = 0; i < frameCount; i++) {
				totalB += frames.at(i).at<cv::Vec3b>(r, c)[0];
				totalG += frames.at(i).at<cv::Vec3b>(r, c)[1];
				totalR += frames.at(i).at<cv::Vec3b>(r, c)[2];
			}

			result.at<cv::Vec3b>(r, c)[0] = totalB / frameCount;
			result.at<cv::Vec3b>(r, c)[1] = totalG / frameCount;
			result.at<cv::Vec3b>(r, c)[2] = totalR / frameCount;
		}
	}

	return result;
}


cv::Mat processForeground(cv::Mat foreground, cv::Mat background, uchar hThresh, uchar sThresh, uchar vThresh)
{
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
int matDiff(cv::Mat mat1, cv::Mat mat2) {
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

	return diff;
}

// Returns the amount of different elements between two binary bitmaps
int matDiffBinary(cv::Mat mat1, cv::Mat mat2) {
	cv::Mat result;
	bitwise_xor(mat1, mat2, result);

	int diff = 0;

	for (int r = 0; r < result.rows; r++) {
		for (int c = 0; c < result.cols; c++) {
			if (result.at<uchar>(r, c) == 255) {
				diff++;
			}
		}
	}

	return diff;
}

cv::Vec3b findHSVThresholds(cv::Mat &reference, cv::Mat &foreground, cv::Mat &background) {
	int step = 10;
	int bestDiff = INT_MAX;
	cv::Vec3b thresholds = (0, 0, 0);

	for (int h = 0; h < 120; h += step) {
		for (int s = 0; s < 120; s += step) {
			for (int v = 0; v < 120; v += step) {
				printf("\n(%d, %d, %d)", h, s, v);
				cv::Mat processed = processForeground(foreground, background, h, s, v);

				cv::waitKey(0);
				int diff = matDiffBinary(reference, processed);
				if (diff < bestDiff) {
					bestDiff = diff;
					thresholds[0] = h; thresholds[1] = s; thresholds[2] = v;
				}
			}
		}
	}

	return thresholds;
}

int main(int argc, char** argv)
{
	//VoxelReconstruction::showKeys();
	//VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	//vr.run(argc, argv);

	trainThresholdValues("data/", "data/gradientDescentOutput.txt");
	cv::waitKey(0);

	return EXIT_SUCCESS;
}