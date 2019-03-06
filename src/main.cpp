#include <cstdlib>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "controllers/Glut.h"
#include "controllers/Reconstructor.h"
#include "controllers/Scene3DRenderer.h"

//#include <climits>

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

cv::Vec3b findHSVThresholds(cv::Mat &reference, cv::Mat &foreground, cv::Mat &background, int step) {
	int bestDiff = INT_MAX;
	cv::Vec3b thresholds = cv::Vec3b(0, 0, 0);

	for (int h = max(0, thresholds[0] - 8 * step); h < min(thresholds[0] + 8 * step, 256); h += step) {
		for (int s = max(0, thresholds[1] - 8 * step); s < min(thresholds[1] + 8 * step, 256); s += step) {
			for (int v = max(0, thresholds[2] - 8 * step); v < min(thresholds[2] + 8 * step, 256); v += step) {
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

	if (step == 1) {
		return thresholds;
	}
	else {
		return findHSVThresholds(reference, foreground, background, step / 2);
	}
}

cv::Vec3b averageColor(cv::Mat foreground, cv::Mat reference, cv::Vec3b referenceColor) {
	int pixelCount = 0;
	cv::Vec3i colors = cv::Vec3i(0, 0, 0);

	for (int r = 0; r < reference.rows; r++) {
		for (int c = 0; c < reference.cols; c++) {
			cv::Vec3b px = reference.at<cv::Vec3b>(r, c);
			if (px[0] == referenceColor[0] && px[1] == referenceColor[1] && px[2] == referenceColor[2]) {
				colors[0] += foreground.at<cv::Vec3b>(r, c)[0];
				colors[1] += foreground.at<cv::Vec3b>(r, c)[1];
				colors[2] += foreground.at<cv::Vec3b>(r, c)[2];
				pixelCount++;
			}
		}
	}

	cv::Vec3b result = cv::Vec3b(0, 0, 0);
	if (pixelCount > 0) {
		result[0] = (uchar)(colors[0] / pixelCount);
		result[1] = (uchar)(colors[1] / pixelCount);
		result[2] = (uchar)(colors[2] / pixelCount);
	}

	return result;
}

int main(int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data/4persons" + std::string(PATH_SEP), 4);
	vr.setParams(64, 10, 4, 0.01);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}