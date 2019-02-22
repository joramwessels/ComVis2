#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"

#include "EdgeDetector.h"

using namespace nl_uu_science_gmt;

void testEdgeDetector()
{
	cv::Mat img = cv::imread("data/cam1/background.png", cv::IMREAD_GRAYSCALE);
	EdgeDetector ed = EdgeDetector();
	double kernel[9] = {
		-1.0, -1.0, -1.0,
		-1.0, 8.0, -1.0,
		-1.0, -1.0, -1.0
	};
	ed.setFilter(3, kernel);
	cv::Mat edges = ed.filterImage(img);
	//cv::Mat edges = ed.threshold(img, 1.0);

	cv::namedWindow("OpenCV");
	cv::imshow("OpenCV", edges);
	cv::waitKey(0);
	printf("Done\n");
}

int main(int argc, char** argv)
{
	testEdgeDetector();
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}