/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <opencv2/core/core.hpp>
#include <opencv2\ml\ml.hpp>
#include <stddef.h>
#include <vector>

#include "Camera.h"

namespace nl_uu_science_gmt
{

class Reconstructor
{
public:
	/*
	 * Voxel structure
	 * Represents a 3D pixel in the half space
	 */
	struct Voxel
	{
		//bool active = false;						// Whether or not the voxel should be drawn
		int x, y, z;								// Coordinates
		cv::Scalar color;							// Color
		std::vector<cv::Point> camera_projection;	// Projection location for camera[c]'s FoV (2D)
		std::vector<int> valid_camera_projection;	// Flag if camera projection is in camera[c]'s FoV
	};

private:
	/* Pointers to all voxels corresponding to a pixel of a specific camera.
	Usage: voxel_pointers[camera][pixel x*y][voxels] */
	std::vector<std::vector<std::vector<Voxel*>>> m_voxel_pointers;

	const std::vector<Camera*> &m_cameras;  // vector of pointers to cameras
	const int m_height;                     // Cube half-space height from floor to ceiling
	const int m_step;                       // Step size (space between voxels)

	std::vector<cv::Point3f*> m_corners;    // Cube half-space corner locations

	size_t m_voxels_amount;                 // Voxel count
	cv::Size m_plane_size;                  // Camera FoV plane WxH

	std::vector<Voxel*> m_voxels;           // Pointer vector to all voxels in the half-space
	std::vector<Voxel*> m_visible_voxels;   // Pointer vector to all visible voxels

	void initialize();
	void initVisibleVoxels();

	// clustering
	void cluster();
	int m_clusterEpochs;
	int m_clusterCount;
	cv::Mat m_cluster_centroids;
	std::vector<int> m_clusterMapping; // maps cluster ID to person ID
	int m_path_scale;
	cv::Mat m_centroid_paths;
	double m_terminationDelta;
	int m_histogramBinCount;
	std::vector<int> m_clusterLabels;
	std::vector<int> m_clusterColors = {
		0xFF0000, 0xFF00, 0xFF,
		0xFFFF00, 0xFFFF, 0xFF00FF
	};
	cv::Vec3b m_avgColorReference[4] = {
		cv::Vec3b(78, 72, 53), cv::Vec3b(43, 38, 26),
		cv::Vec3b(72, 66, 53), cv::Vec3b(85, 72, 50)
	};
	std::vector<std::vector<cv::Mat>> m_histogramReference;

	cv::Vec3b getAverageColor(int clusterIdx);
	std::vector<cv::Mat> getColorHistograms(int clusterIdx, int bins=50);
	
	std::vector<int> findBestAvgColorMatches(std::vector<cv::Vec3b> avgColors);
	std::vector<int> findBestHistogramMatches(std::vector<std::vector<cv::Mat>> histograms);
	std::vector<int> findClosestHistogramReferences(std::vector<std::vector<cv::Mat>> histograms);

public:
	Reconstructor(
			const std::vector<Camera*> &, int voxelStep=64);
	virtual ~Reconstructor();

	void update();

	void initVoxelColoring();

	// Returns which of two voxels should come first in an ordered voxel list
	static bool voxelSort(Voxel &a, Voxel &b) {
		if (a.x != b.x)
			return a.x > b.x;
		else if (a.y != b.y)
			return a.y > b.y;
		else
			return a.z > b.z;
	}

	// Returns whether two voxels are the same
	static bool voxelPred(const Voxel &a, const Voxel &b) {
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}

	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return m_visible_voxels;
	}

	const std::vector<Voxel*>& getVoxels() const
	{
		return m_voxels;
	}

	void setVisibleVoxels(
			const std::vector<Voxel*>& visibleVoxels)
	{
		m_visible_voxels = visibleVoxels;
	}

	void setVoxels(
			const std::vector<Voxel*>& voxels)
	{
		m_voxels = voxels;
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return m_corners;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}

	/*
	Sets clustering parameters
	@param clusterEpochs the number of clustering attempts to average
	@param clusterCount the number of clusters to find
	@param terminationDelta the change in cluster centroid at which to terminate the clustering
	*/
	void setParams(int clusterEpochs, int clusterCount, double terminationDelta, int histogramBinCount)
	{
		assert(clusterCount < 7 && clusterCount > 0);
		m_clusterEpochs = clusterEpochs;
		m_clusterCount = clusterCount;
		m_terminationDelta = terminationDelta;
		m_histogramBinCount = histogramBinCount;
	}

	/*
		Returns a vector with the voxels in the given cluster
		@param clusterIdx the index of the required cluster
	*/
	std::vector<cv::Point3f> getClusterVoxels(int clusterIdx)
	{
		std::vector<cv::Point3f> voxels;
		for (int i = 0; i < m_visible_voxels.size(); i++) if ((int)(m_clusterLabels[i]) == clusterIdx)
			voxels.push_back(cv::Point3f(m_visible_voxels[i]->x, m_visible_voxels[i]->y, m_visible_voxels[i]->z));
		return voxels;
	}

	/*
		Returns the center of a centroid as a point2f (in xy space)
		@param clusterIdx the index of the required cluster
	*/
	cv::Point2f getCentroid(int clusterIdx) {
		float x = m_cluster_centroids.at<float>(clusterIdx, 0);
		float y = m_cluster_centroids.at<float>(clusterIdx, 1);
		return cv::Point2f(x, y);
	}

	void updateCentroidPaths() {
		//cv::Point2f centroid = getCentroid(0);
		//int x = static_cast<int>((centroid.y + m_height) / m_path_scale);
		//int y = static_cast<int>((centroid.x + m_height) / m_path_scale);

		//m_centroid_paths.at<uchar>(y, x) = 255;

		for (int c = 0; c < m_clusterCount; c++) {
			cv::Point2f centroid = getCentroid(c);
			int x = static_cast<int>((centroid.x + m_height) / m_path_scale);
			int y = static_cast<int>((centroid.y + m_height) / m_path_scale);

			m_centroid_paths.at<uchar>(y, x) = 255;
		}
	}

	/*
		prints the histograms as a table for debugging
		@param histograms the histogram vector
	*/
	void printHistograms(std::vector<std::vector<cv::Mat>> histograms)
	{
		printf("\n");
		for (int i = 0; i < 10; i++) printf("%3d ", (256 / 9)*i);
		printf("\n");
		for (int i = 0; i < m_clusterCount; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 10; k++) {
					printf("%3d ", *(histograms[i][j].ptr(k)));
				}
				printf("\n");
			}
		}
	}

	cv::Mat getCentroidPaths() {
		return m_centroid_paths;
	}

	cv::Mat foregroundMask(std::vector<Voxel*> voxels, int cam);
};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
