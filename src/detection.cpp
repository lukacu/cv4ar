
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

#include "marker.h"

using namespace std;
using namespace cv;

#define WINDOW_NAME "Marker detection"

int main(int argc, const char** argv) {

    Mat frame, gray, video_frame, marker;

    string marker_file("marker.png");
    int camera_id = 0;

    if (argc > 1) camera_id = atoi(argv[1]);
    if (argc > 2) marker_file = argv[2];

    marker = imread(marker_file, IMREAD_GRAYSCALE);

    VideoCapture camera(camera_id);

    if (!camera.isOpened()) {
        cout << "Unable to access camera" << endl;
        return -1;
    }

    if (marker.empty()) {
        cout << "Marker not found" << endl;
        return -1;
    }

    while (true) {

        camera.read(frame);

        if (frame.empty())
            break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> points;

        if (detect_template_marker_corners(gray, marker, points)) {

            line(frame, points[0], points[points.size()-1], Scalar(255, 0, 0), 3);

            for (size_t i = 0; i < points.size()-1; i++) {
                line(frame, points[i], points[i+1], Scalar(0, 255, 0), 3);
            }

        }

        imshow(WINDOW_NAME, frame);

        if (waitKey(30) > 0) break;

    }

    return 0;
}

