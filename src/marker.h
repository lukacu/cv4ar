
#ifndef _FEATURES
#define _FEATURES

#include <vector>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

bool detect_template_marker_corners(Mat image, Mat marker, vector<Point2f>& image_points);

bool detect_template_marker(Mat image, Mat pattern, float pattern_size, Mat& homography);

bool detect_template_marker(Mat image, Mat pattern, float pattern_size, Mat intrinsics, Mat distortion, Mat& rotation, Mat& translation);

#endif
