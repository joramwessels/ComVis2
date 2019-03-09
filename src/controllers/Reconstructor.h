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
	double m_terminationDelta;
	std::vector<int> m_clusterLabels;
	cv::Vec3b m_avgColorReference[4] = {
		cv::Vec3b(78, 72, 53), cv::Vec3b(43, 38, 26),
		cv::Vec3b(72, 66, 53), cv::Vec3b(85, 72, 50)
	};
	std::vector<cv::Mat> m_histogramReference;

	cv::Vec3b getAverageColor(int clusterIdx);
	cv::Mat getColorHistogram(int clusterIdx, int bins=10);
	
	std::vector<int> findBestAvgColorMatches(std::vector<cv::Vec3b> avgColors);
	std::vector<int> findBestHistogramMatches(std::vector<cv::Mat> histograms);

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
	void setParams(int clusterEpochs, int clusterCount, double terminationDelta)
	{
		assert(clusterCount < 7 && clusterCount > 0);
		m_clusterEpochs = clusterEpochs;
		m_clusterCount = clusterCount;
		m_terminationDelta = terminationDelta;
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
};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
