/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

#include <opencv2\ml\ml.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>

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
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(auto) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
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
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height) {
						voxel->valid_camera_projection[(int)c] = 1;
						m_voxel_pointers[c][point.x * point.y].push_back(voxel);
					}
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}
	
	cout << "\tdone!" << endl;
}

void Reconstructor::initVoxelColoring() { // TODO never gets called?
	printf("\n\nVoxel count: %d", (int)m_voxels_amount);
	// For every voxel
	for (int v = 0; v < (int)m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
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
	// Initialize visible voxels vector
	if (m_visible_voxels.size() == 0) { initVisibleVoxels(); return; }

	// Voxels that were potentially changed from the last frame to the current.
	// These voxels should be updated to reflect any possible changes.
	std::vector<Voxel*> changed_voxels;
	int cameraCount = (int)m_cameras.size();
	for (int c = 0; c < cameraCount; c++) {
		cv::Mat changed_pixels = m_cameras[c]->getForegroundDifference();
		for (int y = 0; y < changed_pixels.rows; y++) for (int x = 0; x < changed_pixels.cols; x++) {
			if (changed_pixels.at<uchar>(y, x) == 255) {
				changed_voxels.insert(changed_voxels.end(), m_voxel_pointers[c][x * y].begin(), m_voxel_pointers[c][x * y].end());
			}
		}
	}

	int v;
	int changed_voxel_count = (int)changed_voxels.size();
#pragma omp parallel for schedule(auto) private(v) shared(m_visible_voxels)

	// For potentially changed voxels
	for (v = 0; v < changed_voxel_count; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = changed_voxels[v];

		for (int c = 0; c < cameraCount; ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// checking if voxel is already in the visible voxels vector
		int voxIndex = -1, vixVoxCount = m_visible_voxels.size();
		//size_t voxIndex = std::find(m_visible_voxels.begin(), m_visible_voxels.end(), voxel);
		for (int i = 0; i < vixVoxCount; i++) if (m_visible_voxels[i] == voxel) { voxIndex = i; break; }

#pragma omp critical
		// Updating visible voxels vector accordingly
		bool active = (camera_counter == cameraCount), inVector = (voxIndex != -1);
		if (active && !inVector) m_visible_voxels.push_back(voxel);
		if (!active && inVector) m_visible_voxels.erase(m_visible_voxels.begin() + voxIndex);
	}
	cluster();
}

/*
	Clusters the visible voxels and colors them respectively
*/
void Reconstructor::cluster()
{
	assert(m_clusterCount > 0);

	// Collecting voxels
	//printf("Collecting voxels...");
	int N = (int)m_visible_voxels.size();
	std::vector<cv::Point2f> dataPoints;
	for (int i = 0; i < N; i++)
		dataPoints.push_back(cv::Point2f((float)m_visible_voxels[i]->x, (float)m_visible_voxels[i]->y));

	// Clustering voxels
	//printf("Clustering voxels...");
	TermCriteria terminationCriteria = TermCriteria(TermCriteria::EPS, 0, m_terminationDelta);
	cv::Mat centroids = cv::Mat(m_clusterCount, 2, CV_32F);
	for (int i = 0; i < m_clusterCount; i++) centroids.row(i) = cv::Mat(1, 2, CV_32F, { (float)i, 0.0f });
	cv::kmeans(dataPoints, m_clusterCount, m_clusterLabels, terminationCriteria, m_clusterEpochs, KMEANS_PP_CENTERS, centroids);

	// Matching clusters with reference color models
	if (m_histogramReference.size() == 0)
		for (int i = 0; i < m_clusterCount; i++)
			m_histogramReference.push_back(getColorHistogram(i));
	std::vector<cv::Mat> histograms = std::vector<cv::Mat>(m_clusterCount);
	for (int i = 0; i < m_clusterCount; i++) histograms[i] = getColorHistogram(i);
	std::vector<int> clusterIdx = findBestHistogramMatches(histograms);

	// Coloring voxels
	//printf("Coloring voxels...\n");
	uint color;
	for (int i = 0; i < N; i++)
	{
		// TODO cluster doesn't retain same color
		if (clusterIdx[m_clusterLabels[i]] == 0) color = 0xFF0000;
		else if (clusterIdx[m_clusterLabels[i]] == 1) color = 0xFF00;
		else if (clusterIdx[m_clusterLabels[i]] == 2) color = 0xFF;
		else if (clusterIdx[m_clusterLabels[i]] == 3) color = 0xFFFF00;
		else if (clusterIdx[m_clusterLabels[i]] == 4) color = 0x00FFFF;
		else if (clusterIdx[m_clusterLabels[i]] == 5) color = 0xFF00FF;
		*((uint*)(&(m_visible_voxels[i]->color))) = color; // got tired of casting cv::Scalar to GLfloat
	}

	//for (int i = 0; i < m_clusterCount; i++) // DEBUG
	//	printf("centroid%i: %.2f, %.2f 0.0\n", i,
	//		centroids.at<float>(i, 0),
	//		centroids.at<float>(i, 1));
}

/*
	Computes the average color of the projected voxel cluster over all cameras
	@param clusterIdx the cluster for which to get the average color
*/
cv::Vec3b Reconstructor::getAverageColor(int clusterIdx)
{
	std::vector<cv::Vec3b> pixels;
	int voxelCount = (int)m_visible_voxels.size(), cameraCount = m_cameras.size();
//#pragma omp parallel for schedule (auto) private(i) shared(pixels)
	for (int i = 0; i < voxelCount; i++) if (m_clusterLabels[i] == clusterIdx) {
		for (int c = 0; c < cameraCount; c++)
		{
			cv::Point pixId = m_visible_voxels[i]->camera_projection[c];
			cv::Mat frame = m_cameras[c]->getFrame();
//#pragma omp critical
			pixels.push_back(frame.at<Vec3b>(pixId.y, pixId.x));
		}
	}
	cv::Vec3i sum;
	int pixelCount = pixels.size();
	for (int i = 0; i < pixelCount; i++) sum += pixels[i];
	return cv::Vec3b(sum[0] / pixelCount, sum[1] / pixelCount, sum[2] / pixelCount);
}

/*
	Calculates the color histogram for the given cluster ID
	@param clusterIdx the cluster ID to calculate a histogram for
	@param bins the amount of color bins for each channel
*/
cv::Mat Reconstructor::getColorHistogram(int clusterIdx, int binCount)
{
	std::vector<cv::Vec3b> values;
	int voxelCount = (int)m_visible_voxels.size(), cameraCount = (int)m_cameras.size();
	for (int c = 0; c < cameraCount; c++)
	{
		// Collecting projected pixels
		std::vector<Point> pixels;
		for (int i = 0; i < voxelCount; i++) if (m_clusterLabels[i] == clusterIdx) {
			cv::Point pix = m_visible_voxels[i]->camera_projection[c];
			//for (int j=0; j<pixels.size(); j++) if (pixels[j].x == pix.x && pixels[j].y == pix.y)
				pixels.push_back(pix); break;
		}

		// Collecting corresponding values
		int pixelCount = pixels.size();
		for (int i = 0; i < pixelCount; i++) {
			values.push_back(m_cameras[c]->getFrame().at<Vec3b>(pixels[i]));
		}
	}

	int binSize = 256 / binCount; // TODO this actually has one bin more than binCount
	int valueCount = values.size();
	cv::Mat histogram = cv::Mat::zeros(binCount+1, binCount+1, CV_32S);
	for (int i = 0; i < valueCount; i++)
	{
		int Hbin = values[i][0] / binSize;
		int Sbin = values[i][1] / binSize;
		histogram.at<int>(Hbin, Sbin) = histogram.at<int>(Hbin, Sbin) + 1;
	}
	return histogram.reshape(1, 1);
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
	Finds the best matching color model using histograms
	@param histograms a vector with the color histograms
*/
std::vector<int> Reconstructor::findBestHistogramMatches(std::vector<cv::Mat> histograms)
{
	// Finding the closest reference per cluster (Euclidean)
	std::vector<int> bestMatch;
	std::vector<float> closestDist;
	//cv::Mat dist = cv::Mat(m_clusterCount, m_clusterCount, CV_32F);
	for (int i = 0; i < m_clusterCount; i++)
	{
		bestMatch.push_back(-1);
		closestDist.push_back(-1.0f);
		for (int j = 0; j < m_clusterCount; j++)
		{
			cv::Mat err = m_histogramReference[j] - histograms[i];
			float sqrDist = err.dot(err);
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

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
		}
	}

	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());
}

} /* namespace nl_uu_science_gmt */
