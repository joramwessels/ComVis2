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

	for (int h = std::max(0, thresholds[0] - 8 * step); h < std::min(thresholds[0] + 8 * step, 256); h += step) {
		for (int s = std::max(0, thresholds[1] - 8 * step); s < std::min(thresholds[1] + 8 * step, 256); s += step) {
			for (int v = std::max(0, thresholds[2] - 8 * step); v < std::min(thresholds[2] + 8 * step, 256); v += step) {
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

int main(int argc, char** argv)
{
	// ORIGINAL CODE
	//VoxelReconstruction::showKeys();
	//VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	//vr.run(argc, argv);

	const char* filename = "data/gradientDescentOutput.txt";
	int rotationSpeed = 30;

	// Determining thresholds
	std::vector<cv::Vec3f> thresholds(4);
	printf("Reading threshold values from %s\n", filename);
	if (!readThresholds(filename, thresholds))
	{
		printf("No valid threshold file found. Commencing threshold training...");
		thresholds = trainThresholdValues("data/", filename);
	}
	cv::Vec3b avg;
	avg[0] = (uchar)((thresholds[0][0] + thresholds[1][0] + thresholds[2][0] + thresholds[3][0]) / 4);
	avg[1] = (uchar)((thresholds[0][1] + thresholds[1][1] + thresholds[2][1] + thresholds[3][1]) / 4);
	avg[2] = (uchar)((thresholds[0][2] + thresholds[1][2] + thresholds[2][2] + thresholds[3][2]) / 4);
	printf("Thresholds:\nH: %i\nS: %i\nV: %i\n", avg[0], avg[1], avg[2]);

	// Initializing cameras
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	std::vector<nl_uu_science_gmt::Camera*> cameras = vr.getCameras();
	for (int v = 0; v < 4; ++v)
	{
		bool has_cam = Camera::detExtrinsics(cameras[v]->getDataPath(), General::CheckerboadVideo,
			General::IntrinsicsFile, cameras[v]->getCamPropertiesFile());
		assert(has_cam);
		cameras[v]->initialize();
	}

	// Presenting results in 3D scene
	cv::namedWindow(VIDEO_WINDOW, CV_WINDOW_KEEPRATIO);
	nl_uu_science_gmt::Reconstructor reconstructor(cameras);
	nl_uu_science_gmt::Scene3DRenderer scene3d(reconstructor, cameras);
	nl_uu_science_gmt::Glut glut(scene3d);
	glut.initializeWindows(SCENE_WINDOW.c_str());
	while (!scene3d.isQuit())
	{
		glut.motion(rotationSpeed, 0);
		glut.update(0);
		glut.display();
	}
	return EXIT_SUCCESS;
}