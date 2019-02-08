
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

#define WINDOW_NAME "Homography projection"
#define MARKER_SIZE 200

void warpImage(Mat& target, Mat& image, const Mat& transformation, int flags = INTER_LINEAR) {

    Mat transformed, alpha;
    cv::warpPerspective(image, transformed, transformation, target.size(), flags);
    cv::warpPerspective(Mat::ones(image.size(), CV_32FC1), alpha, transformation, target.size(), flags);

    uchar* target_ptr = reinterpret_cast<uchar*>(target.data);
    uchar* transformed_ptr = reinterpret_cast<uchar*>(transformed.data);
    float* alpha_ptr = reinterpret_cast<float*>(alpha.data);

    for(int i = 0; i < target.rows * target.cols; i++, alpha_ptr++) {
        for (int j = 0; j < target.channels(); j++, target_ptr++, transformed_ptr++)
            *target_ptr = (uchar) ((float)(*transformed_ptr) * (*alpha_ptr) + (float)(*target_ptr) * (1 - *alpha_ptr));
    }

}


int main(int argc, const char** argv) {

    Mat frame, gray, video_frame, marker;

    string marker_file("marker.png");
    string video_file("video.mp4");
    int camera_id = 0;

    if (argc > 1) camera_id = atoi(argv[1]);
    if (argc > 2) marker_file = argv[2];
    if (argc > 3) video_file = argv[3];

    marker = imread(marker_file, IMREAD_GRAYSCALE);

    VideoCapture camera(camera_id);
    VideoCapture video(video_file);

    if (!camera.isOpened()) {
        cout << "Unable to access camera" << endl;
        return -1;
    }

    if (!video.isOpened()) {
        cout << "Unable to access video file" << endl;
        return -1;
    }

    if (marker.empty()) {
        cout << "Marker not found" << endl;
        return -1;
    }

    Mat homography;

    while (true) {

        video.read(video_frame);

        if (video_frame.empty()) {
            video.set(CAP_PROP_POS_FRAMES, 0);
            video.read(video_frame);
        }

        camera.read(frame);

        if (frame.empty())
            break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        bool positioned = detect_template_marker(gray, marker, MARKER_SIZE, homography);

        if (positioned)
            warpImage(frame, video_frame, homography);

        imshow(WINDOW_NAME, frame);

        if (waitKey(30) > 0) break;

    }

    return 0;
}

