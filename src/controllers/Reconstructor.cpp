/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

#include <opencv2\opencv.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>
#include <chrono>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs, int voxelStep) :
				m_cameras(cs),
				m_height(2048),
				m_step(voxelStep)
{
	m_voxel_pointers.resize(m_cameras.size());
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();

		m_voxel_pointers[c].resize(m_plane_size.width * m_plane_size.height);
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	m_path_scale = 4;
	m_centroid_paths = cv::Mat::zeros(edge / m_path_scale, edge / m_path_scale, CV_8UC3);
	m_centroid_paths.setTo(cv::Scalar(200, 200, 200));

	Reconstructor::getSize();

	initialize();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	cout << "Voxel size: " << sizeof(Voxel) << endl;

	time_t init_start_time = time(0);

	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels: ";
	// m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

		if (done > pdone)
		{
			pdone = done;
#ifdef _WIN32
			printf("%2d%%\b\b\b", done);
#else
			cout << done << "%..." << flush;
#endif
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				// voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				vector<Point> camera_projections;
				camera_projections.resize(4);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				bool valid = true;
				for (int c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height) {
						// voxel->valid_camera_projection[(int)c] = 1;
						camera_projections[c] = point;
						// m_voxel_pointers[c][(point.y * m_plane_size.width) + point.x].push_back(voxel);
					}
					else {
						valid = false;
						break;
					}
				}

				// Writing voxel 'p' is not critical as it's unique (thread safe)
				if (valid) {
					for (int c = 0; c < m_cameras.size(); ++c) {
						Point point = camera_projections[c];
						m_voxel_pointers[c][(point.y * m_plane_size.width) + point.x].push_back(voxel);
					}
					m_voxels.push_back(voxel);
				}
				// m_voxels[p] = voxel;
			}
		}
	}

	m_voxels_amount = m_voxels.size();

	time_t init_end_time = time(0);
	cout << "\n\nInitialization time: " << init_end_time - init_start_time << "s" << endl;
	cout << "Initialized " << m_voxels.size() << " voxels" << endl;
}

void Reconstructor::initVoxelColoring() { // TODO never gets called?
	printf("\n\nVoxel count: %d\n", (int)m_voxels_amount);
	// For every voxel
	for (int v = 0; v < (int)m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			const Point point = voxel->camera_projection[c];

			//If there's a white pixel on the foreground image at the projection point, add the camera
			if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
		}

		//voxel->active = (camera_counter == m_cameras.size());
	}
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	int mode = 3;
	// 0: diff
	// 1: camera 0 foreground
	// 2: loop over voxels
	// 3: GPGPU

	// Initialize visible voxels vector
	if (m_visible_voxels.size() == 0) { initVisibleVoxels(); return; }

	if (frame_count == 0) {
		cout << "\nTime started" << endl;
		start_time = time(0);
	}
	frame_count++;

	if (frame_count == 500) {
		time_t end_time = time(0);
		cout << "\nElapsed time: " << end_time - start_time << "s" << endl;
		cout << "Average FPS:  " << 500 / (end_time - start_time) << endl;
	}

	if (mode == 3) {
		for (int c = 0; c < m_cameras.size(); c++) {
			cv::Mat diff_pixels = m_cameras[c]->getDiffPixels();

			// For every different pixels in this camera's view
			for (int y = 0; y < diff_pixels.rows; y++) for (int x = 0; x < diff_pixels.cols; x++) {
				if (diff_pixels.at<uchar>(y, x) == 255) {

					// For every voxel corresponding to that pixel
					for (int v = 0; v < m_voxel_pointers[c][(y * m_plane_size.width) + x].size(); v++) {

						// If the voxel is active, check if it should be inactive
						if (m_voxel_pointers[c][(y * m_plane_size.width) + x][v]->active) {
							for (int c2 = 0; c2 < m_cameras.size(); c2++) {
								const Point point = m_voxel_pointers[c][(y * m_plane_size.width) + x][v]->camera_projection[c2];
								
								if (m_cameras[c2]->getForegroundImage().at<uchar>(point) == 0) {
									m_voxel_pointers[c][(y * m_plane_size.width) + x][v]->active = false;
								}
							}
						}
						// If it's not active, check if it should be active
						else {
							bool active = true;
							for (int c2 = 0; c2 < m_cameras.size(); c2++) {
								const Point point = m_voxel_pointers[c][(y * m_plane_size.width) + x][v]->camera_projection[c2];
								if (m_cameras[c2]->getForegroundImage().at<uchar>(point) == 0) {
									active = false;
									break;
								}
							}
							if (active) {
								m_voxel_pointers[c][(y * m_plane_size.width) + x][v]->active = true;
							}
						}
					}
				}
			}
		}
	}

	// -------------------------------------------------------------------------------------------------------------------------

	if (mode == 0) {
		//// For every pixel that went from active to inactive in the last frame, set their corresponding voxels to inactive
		//for (int c = 0; c < m_cameras.size(); c++) {
		//	cv::Mat old_pixels = m_cameras[c]->getOldPixels();
		//	for (int y = 0; y < old_pixels.rows; y++) for (int x = 0; x < old_pixels.cols; x++) {
		//		if (old_pixels.at<uchar>(y, x) == 255) {
		//			for (int v = 0; v < m_voxel_pointers[c][(y * m_plane_size.width) + x].size(); v++) {
		//				m_voxel_pointers[c][(y * m_plane_size.width) + x][v]->active = false;
		//			}
		//		}
		//	}
		//}

		//// Remove all inactive voxels from visible voxels vector
		//for (int i = 0; i < m_visible_voxels.size(); i++) {
		//	if (!m_visible_voxels[i]->active) {
		//		m_visible_voxels.erase(m_visible_voxels.begin() + i);
		//	}
		//}

		for (int c = 0; c < m_cameras.size(); c++) {
			cv::Mat foreground = m_cameras[c]->getForegroundImage();

			for (int i = 0; i < m_visible_voxels.size(); i++) {
				const Point point = m_visible_voxels[i]->camera_projection[c];
				if (foreground.at<uchar>(point) == 0) {
					m_visible_voxels[i]->active = false;
				}
			}
		}

		// For every pixel that went from inactive to active, check if their voxels are flagged as inactive
		// then check if they should be active and add them to the visible voxel vector where necessary
		for (int c = 0; c < m_cameras.size(); c++) {
			cv::Mat new_pixels = m_cameras[c]->getNewPixels();
			
			// For every new pixels in this camera's view
			for (int y = 0; y < new_pixels.rows; y++) for (int x = 0; x < new_pixels.cols; x++) {
				if (new_pixels.at<uchar>(y, x) == 255) {

					// For every voxel corresponding to that pixel
					for (int v = 0; v < m_voxel_pointers[c][(y * m_plane_size.width) + x].size(); v++) {

						// If it's not active, check if it should be active
						if (!m_voxel_pointers[c][(y * m_plane_size.width) + x][v]->active) {
							bool active = true;
							for (int c2 = 0; c2 < m_cameras.size(); c2++) {
								const Point point = m_voxel_pointers[c][(y * m_plane_size.width) + x][v]->camera_projection[c2];
								if (m_cameras[c2]->getForegroundImage().at<uchar>(point) == 0) {
									active = false;
									break;
								}
							}
							if (active) {
								m_voxel_pointers[c][(y * m_plane_size.width) + x][v]->active = true;
							}
						}
					}
				}
			}
		}
		
		m_visible_voxels.clear();
		for (int v = 0; v < (int)m_voxels_amount; ++v) {
			Voxel* voxel = m_voxels[v];

			if (voxel->active) {
				m_visible_voxels.push_back(voxel);
			}
		}
	}

	// -------------------------------------------------------------------------------------------------------------------------

	if (mode == 1) {
		m_visible_voxels.clear();

		cv::Mat active_pixels = m_cameras[0]->getForegroundImage();
		for (int y = 0; y < m_plane_size.height; y++) for (int x = 0; x < m_plane_size.width; x++) {
			if (active_pixels.at<uchar>(y, x) == 255) {
				for (int v = 0; v < m_voxel_pointers[0][(y * m_plane_size.width) + x].size(); v++) {
					bool active = true;
					for (int c = 1; c < m_cameras.size(); c++) {
						const Point point = m_voxel_pointers[0][(y * m_plane_size.width) + x][v]->camera_projection[c];
						if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 0) {
							active = false;
							break;
						}
					}
					if (active) {
						m_visible_voxels.push_back(m_voxel_pointers[0][(y * m_plane_size.width) + x][v]);
					}
				}
			}
		}
	}

	// -------------------------------------------------------------------------------------------------------------------------

	if (mode == 2) {
		initVisibleVoxels();
	}
}

/*

	@param voxels
	@param camNum
*/
cv::Mat Reconstructor::foregroundMask(std::vector<Voxel*> voxels, int camNum) {
	cv::Mat mask = cv::Mat(m_cameras[camNum]->getSize(), CV_8U);

	for (int v = 0; v < voxels.size(); v++) {
		int y = voxels[v]->camera_projection[camNum].y;
		int x = voxels[v]->camera_projection[camNum].x;
		mask.at<uchar>(y, x) = 255;
	}

	return mask;
}

/*
	Clusters the visible voxels and colors them respectively
*/
void Reconstructor::cluster()
{
	assert(m_clusterCount > 0);

	// Collecting voxels
	int N = (int)m_visible_voxels.size();
	std::vector<cv::Point2f> dataPoints;
	for (int i = 0; i < N; i++)
		dataPoints.push_back(cv::Point2f((float)m_visible_voxels[i]->x, (float)m_visible_voxels[i]->y));

	// Clustering voxels
	TermCriteria terminationCriteria = TermCriteria(TermCriteria::EPS, 0, m_terminationDelta);
	cv::Mat centroids = cv::Mat(m_clusterCount, 2, CV_32F);
	for (int i = 0; i < m_clusterCount; i++) centroids.row(i) = cv::Mat(1, 2, CV_32F, { (float)i, 0.0f });
	cv::kmeans(dataPoints, m_clusterCount, m_clusterLabels, terminationCriteria, m_clusterEpochs, KMEANS_PP_CENTERS, centroids);
	m_cluster_centroids = centroids;

	// Matching clusters with reference color models
	if (m_histogramReference.size() == 0)
		for (int i = 0; i < m_clusterCount; i++)
			m_histogramReference.push_back(getColorHistograms(i, m_histogramBinCount));
	std::vector<std::vector<cv::Mat>> histograms = std::vector<std::vector<cv::Mat>>(m_clusterCount);
	for (int i = 0; i < m_clusterCount; i++) histograms[i] = getColorHistograms(i, m_histogramBinCount);
	m_clusterMapping = findBestHistogramMatches(histograms);

	// Coloring voxels
	uint color; // 0x00RRGGBB
	for (int i = 0; i < N; i++)
	{
		// Takes the color of the cluster reference corresponding to that cluster label
		//	(i.e. voxel cluster label --> person ID --> color)
		color = m_clusterColors[m_clusterMapping[m_clusterLabels[i]]];
		// *((uint*)(&(m_visible_voxels[i]->color))) = color; // got tired of casting cv::Scalar to GLfloat
	}
	updateCentroidPaths();
}

/*
	Calculates the color histograms for the given cluster ID
	@param clusterIdx the cluster ID to calculate a histogram for
	@param bins the amount of color bins for each channel
*/
std::vector<cv::Mat> Reconstructor::getColorHistograms(int clusterIdx, int binCount)
{
	// For each camera
	std::vector<cv::Vec3b> values;
	int voxelCount = (int)m_visible_voxels.size(), cameraCount = (int)m_cameras.size();
	for (int c = 0; c < cameraCount; c++)
	{
		// Collect projected pixels without duplicates
		std::vector<Point> pixels;
		for (int i = 0; i < voxelCount; i++) if (m_clusterLabels[i] == clusterIdx) {
			cv::Point pix = m_visible_voxels[i]->camera_projection[c];
			//for (int j=0; j<pixels.size(); j++) if (pixels[j].x != pix.x && pixels[j].y != pix.y) // checking for duplicates
				pixels.push_back(pix);// break;
		}

		// Collect corresponding HSV pixel values
		int pixelCount = pixels.size();
		cv::Mat frame =  m_cameras[c]->getFrame(), hsvFrame;
		cv::cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
		for (int i = 0; i < pixelCount; i++) {
			values.push_back(hsvFrame.at<Vec3b>(pixels[i]));
		}
	}

	// Create two 1-dimensional histograms (for H and S)
	int binSize = 256 / binCount;
	int valueCount = values.size();
	cv::Mat histogramH = cv::Mat::zeros(binCount+2, 1, CV_32F);
	cv::Mat histogramS = cv::Mat::zeros(binCount+2, 1, CV_32F);
	for (int i = 0; i < valueCount; i++)
	{
		int Hbin = values[i][0] / binSize;
		(*(histogramH.ptr(Hbin)))++;
		int Sbin = values[i][1] / binSize;
		(*(histogramS.ptr(Sbin)))++;
	}
	return std::vector<cv::Mat>({ histogramH, histogramS });
}

/*
	Finds the best matching color model using histograms
	@param histograms a vector with the color histograms
*/
std::vector<int> Reconstructor::findBestHistogramMatches(std::vector<std::vector<cv::Mat>> histograms)
{
	//printHistograms(histograms); // DEBUG

	// Collecting all permutations
	std::vector<std::vector<int>> clusterMapping; // maps cluster centroid IDs to person IDs
	std::vector<int> permutation = std::vector<int>(m_clusterCount);
	for (int i = 0; i < m_clusterCount; i++) permutation[i] = i;
	do { clusterMapping.push_back(permutation); }
	while (std::next_permutation(permutation.begin(), permutation.end()));
	int permutationCount = clusterMapping.size();

	// Exhaustively computing the distance for each mapping
	int bestMappingIndex = -1;
	float smallestError = -1.0f;
	float highestError = -1.0f; // DEBUG
	#pragma omp parallel for schedule(auto) shared(bestMappingIndex, smallestError, highestError)
	for (int i = 0; i < permutationCount; i++) // for each permutation
	{
		double summedError = 0;
		for (int j = 0; j < m_clusterCount; j++) // for each cluster
		{
			float dist1 = cv::EMDL1(m_histogramReference[clusterMapping[i][j]][0], histograms[j][0]);
			//cv::Mat err1 = m_histogramReference[clustAssignment[j]][0] - histograms[j][0];
			//float dist1 = err1.dot(err1);
			float dist2 = cv::EMDL1(m_histogramReference[clusterMapping[i][j]][1], histograms[j][1]);
			//cv::Mat err2 = m_histogramReference[clustAssignment[j]][1] - histograms[j][1];
			//float dist2 = err1.dot(err1);
			//printf("%i, %i: (%.2f, %.2f)\n", clustAssignment[j], j, dist1, dist2); // DEBUG
			summedError += dist1 + dist2;
		}
		summedError *= 10000000000000000000000000000000000000000.0;
		//printf("Permutation %i: (%i, %i, %i, %i), err: %f\n", i, clusterMapping[i][0], clusterMapping[i][1], clusterMapping[i][2], clusterMapping[i][3], summedError); // DEBUG
		#pragma omp critical
		if (bestMappingIndex == -1 || summedError < smallestError)
		{
			bestMappingIndex = i;
			smallestError = summedError;
		}
		if (summedError > highestError) highestError = summedError; // DEBUG
	}

	//printf("Best index: %i, lowest: %f, highest %f\n", bestMappingIndex, smallestError, highestError); // DEBUG
	//printf("\nBest: err = %.2f\n", sumOfSquares[bestIndex]); // DEBUG
	//for (int i = 0; i < m_clusterCount; i++) printf("%i, %i: (%.2f)\n", clustAssignment[i], i); // DEBUG

	return clusterMapping[bestMappingIndex];
}

/*
	The simplest histogram matching algorithm, finding the closest reference for each cluster
	@param histograms a vector with the color histograms
*/
std::vector<int> Reconstructor::findClosestHistogramReferences(std::vector<std::vector<cv::Mat>> histograms)
{
	std::vector<int> bestMatch;
	std::vector<float> closestDist;
	for (int i = 0; i < m_clusterCount; i++) // for every cluster
	{
		bestMatch.push_back(-1);
		closestDist.push_back(-1.0f);
		for (int j = 0; j < m_clusterCount; j++) // for every reference
		{
			cv::Mat err1 = m_histogramReference[j][0] - histograms[i][0];
			cv::Mat err2 = m_histogramReference[j][1] - histograms[i][1];
			float sqrDist = err1.dot(err1) + err2.dot(err2);
			if (closestDist[i] == -1.0 || sqrDist < closestDist[i])
			{
				bestMatch[i] = j;
				closestDist[i] = sqrDist;
			}
		}
	}
	return bestMatch;
}

/*
Computes the average color of the projected voxel cluster over all cameras
@param clusterIdx the cluster for which to get the average color
*/
cv::Vec3b Reconstructor::getAverageColor(int clusterIdx)
{
	std::vector<cv::Vec3b> pixels;
	int voxelCount = (int)m_visible_voxels.size(), cameraCount = m_cameras.size();
#pragma omp parallel for schedule (auto) private(i) shared(pixels)
	for (int i = 0; i < voxelCount; i++) if (m_clusterLabels[i] == clusterIdx) {
		for (int c = 0; c < cameraCount; c++)
		{
			cv::Point pixId = m_visible_voxels[i]->camera_projection[c];
			cv::Mat frame = m_cameras[c]->getFrame();
#pragma omp critical
			pixels.push_back(frame.at<Vec3b>(pixId.y, pixId.x));
		}
	}
	cv::Vec3i sum;
	int pixelCount = pixels.size();
	for (int i = 0; i < pixelCount; i++) sum += pixels[i];
	return cv::Vec3b(sum[0] / pixelCount, sum[1] / pixelCount, sum[2] / pixelCount);
}

/*
Finds the best matching color model using average color
@param avgColor the average color of the voxel cluster projections
*/
std::vector<int> Reconstructor::findBestAvgColorMatches(std::vector<cv::Vec3b> avgColors)
{
	// Finding the mapping with the least squares (Euclidean)
	std::vector<int> clustAssignment = std::vector<int>(m_clusterCount);
	for (int i = 0; i < m_clusterCount; i++) clustAssignment[i] = i;

	int i = 0;
	std::vector<float> sumOfSquares;
	while (std::next_permutation(clustAssignment.begin(), clustAssignment.end()))
	{
		sumOfSquares.push_back(0);
		for (int j = 0; j < m_clusterCount; j++)
		{
			cv::Vec3i err = avgColors[j] - m_avgColorReference[clustAssignment[j]];
			sumOfSquares[i] += err.dot(err);
		}
		i++;
	}
	int bestIndex = std::distance(sumOfSquares.begin(), std::min_element(sumOfSquares.begin(), sumOfSquares.end()));
	for (int i = 0; i < m_clusterCount; i++) clustAssignment[i] = i;
	for (int i = 0; i < bestIndex; i++) std::next_permutation(clustAssignment.begin(), clustAssignment.end());
	return clustAssignment;

	//// Finding the closest reference per cluster (Euclidean)
	//int bestMatch = -1;
	//float closestDist = -1.0;
	//for (int i = 0; i < m_clusterCount; i++)
	//{
	//	cv::Vec3b err = cv::Vec3b(avgColor[0] - m_avgColorReference[i][0], avgColor[1] - m_avgColorReference[i][1], avgColor[2] - m_avgColorReference[i][2]);
	//	float length = sqrt(err.dot(err));
	//	if (closestDist == -1.0 || length < closestDist)
	//	{
	//		bestMatch = i;
	//		closestDist = length;
	//	}
	//}
	//return bestMatch;
}

/*
	The original update implementation, now only called on the first frame
*/
void Reconstructor::initVisibleVoxels()
{
	m_visible_voxels.clear();
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for schedule(auto) private(v) shared(visible_voxels)

	// For every voxel
	for (v = 0; v < (int) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		// For every camera
		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			const Point point = voxel->camera_projection[c];

			//If there's a white pixel on the foreground image at the projection point, add the camera
			if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 0) break;
			camera_counter++;
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			voxel->active = true;
			visible_voxels.push_back(voxel);
		}
	}

	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());
}

} /* namespace nl_uu_science_gmt */
