#include "PCH.h"

using namespace nl_uu_science_gmt;

int main(int argc, char** argv)
{
	//cv::Mat testMat = cv::imread("data/test/testimage2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//cv::imshow("test", testMat);
	//cv::waitKey(0);

	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data/4persons" + std::string(PATH_SEP), 4);

	vr.setParams(16);
	vr.setHSVThresholds(5, 10, 50);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}