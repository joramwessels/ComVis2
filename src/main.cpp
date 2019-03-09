#include <cstdlib>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "controllers/Glut.h"
#include "controllers/Reconstructor.h"
#include "controllers/Scene3DRenderer.h"

#include <climits>

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

// Erodes a bitmap image.
// Types: 0 = rectangle, 1 = cross, 2 = ellipse
// Actual size: 2 * size + 1
cv::Mat erodeBitmap(cv::Mat bitmap, int type, int size, int repeat = 1, bool show = false) {
	cv::Mat result = bitmap.clone();
	cv::Mat element = cv::getStructuringElement(type, cv::Size(2 * size + 1, 2 * size + 1), cv::Point(size, size));

	for (int i = 0; i < repeat; i++) {
		erode(result, result, element);
	}

	if (show) {
		cv::imshow("Erosion result", result);
		cv::waitKey(0);
	}

	return result;
}

// Dilates a bitmap image.
// Types: 0 = rectangle, 1 = cross, 2 = ellipse
// Actual size: 2 * size + 1
cv::Mat dilateBitmap(cv::Mat bitmap, int type, int size, int repeat = 1, bool show = false) {
	cv::Mat result = bitmap.clone();
	cv::Mat element = cv::getStructuringElement(type, cv::Size(2 * size + 1, 2 * size + 1), cv::Point(size, size));

	for (int i = 0; i < repeat; i++) {
		dilate(result, result, element);
	}

	if (show) {
		cv::imshow("Dilation result", result);
		cv::waitKey(0);
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

	fg = dilateBitmap(fg, cv::MORPH_ELLIPSE, 2);
	fg = erodeBitmap(fg, cv::MORPH_CROSS, 6);
	fg = dilateBitmap(fg, cv::MORPH_ELLIPSE, 4);

	return fg;
}

// Returns the amount of different elements in two equally sized matrices
int matDiffCount(cv::Mat mat1, cv::Mat mat2) {
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
int matDiffCountBinary(cv::Mat mat1, cv::Mat mat2) {
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
				int diff = matDiffCountBinary(reference, processed);
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

cv::Mat foregroundDiff(cv::Mat firstMat, cv::Mat secondMat, cv::Mat mask) {
	cv::Mat result;
	bitwise_xor(firstMat, secondMat, result, mask);
	return result;
}

int main(int argc, char** argv)
{
	//cv::Vec3b thresholds = cv::Vec3b{ 5, 23, 52 };

	//for (int i = 1; i <= 4; ++i) {
	//	cv::Mat foreground = cv::imread("data/4persons/cam" + std::to_string(i) + "/foreground.png");
	//	cv::Mat background = cv::imread("data/4persons/cam" + std::to_string(i) + "/background.png");
	//	cv::Mat reference = cv::imread("data/4persons/cam" + std::to_string(i) + "/reference.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//	cv::Vec3b thresholds = findHSVThresholds(reference, foreground, background, 16);
	//	printf("Thresholds: (%d, %d, %d)", thresholds[0], thresholds[1], thresholds[2]);
	//	cv::Mat processed = processForeground(foreground, background, thresholds[0], thresholds[1], thresholds[2]);
	//	processed = dilateBitmap(processed, cv::MORPH_ELLIPSE, 2, 1, true);
	//	processed = erodeBitmap(processed, cv::MORPH_CROSS, 6, 1, true);
	//	processed = dilateBitmap(processed, cv::MORPH_ELLIPSE, 4, 1, true);
	//}

	//getchar();

	//std::vector<std::vector<std::vector<float>>> test;

	//cv::Mat testMat1 = cv::imread("data/test/testimage1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat testMat2 = cv::imread("data/test/testimage2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat testMask = cv::imread("data/test/mask.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	//cv::Mat diff = foregroundDiff(testMat1, testMat2, testMask);

	//cv::imshow("Difference", diff);
	//cv::waitKey(0);

	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data/4persons" + std::string(PATH_SEP), 4);
	vr.setParams(64, 10, 4, 0.01); // passing clustering parameters
	vr.setHSVThresholds(5, 10, 50);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}