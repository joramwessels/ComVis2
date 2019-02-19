#pragma once

#ifndef PCH_H
#define PCH_H

// Suppressing file IO deprecation warnings
#define _CRT_SECURE_NO_WARNINGS

// C/C++
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cassert>
#include <stddef.h>
#include <fstream>
#include <cmath>
#include <complex>
#include <string>
#include <valarray>
#include <vector>
#include <cassert>

// OS
#include <Windows.h>
#include <GL/gl.h>
#include <GL/glu.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

// Given code
#include "controllers/arcball.h"
#include "controllers/Camera.h"
#include "controllers/Glut.h"
#include "controllers/Reconstructor.h"
#include "controllers/Scene3DRenderer.h"

#include "utilities/General.h"

#include "VoxelReconstruction.h"

#endif //PCH_H
