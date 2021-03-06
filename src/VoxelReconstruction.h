/*
 * VoxelReconstruction.h
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#ifndef VOXELRECONSTRUCTION_H_
#define VOXELRECONSTRUCTION_H_

#include <string>
#include <vector>

#include "controllers/Camera.h"

namespace nl_uu_science_gmt
{

class VoxelReconstruction
{
	const std::string m_data_path;
	const int m_cam_views_amount;

	std::vector<Camera*> m_cam_views;

	int m_voxelStepSize = 32;
	int m_clusterEpochs = 10;
	int m_clusterCount = 4;
	double m_terminationDelta = 0.01;
	int m_histogramBinCount = 20;

	int m_h_thrs = 20;
	int m_s_thrs = 20;
	int m_v_thrs = 20;

public:
	VoxelReconstruction(const std::string &, const int);
	virtual ~VoxelReconstruction();

	static void showKeys();

	void run(int, char**);

	std::vector<Camera*> getCameras() const { return m_cam_views; }

	/*
		Sets clustering parameters
		@param voxelStepSize the space between each voxel
		@param clusterEpochs the number of clustering attempts to average
		@param clusterCount the number of clusters to find
		@param terminationDelta the change in cluster centroid at which to terminate the clustering
	*/
	void setParams(int voxelStepSize)
	{
		m_voxelStepSize = voxelStepSize;
	}

	void setHSVThresholds(int h, int s, int v) { m_h_thrs = s; m_s_thrs = s; m_v_thrs = v; }
};

} /* namespace nl_uu_science_gmt */

#endif /* VOXELRECONSTRUCTION_H_ */
