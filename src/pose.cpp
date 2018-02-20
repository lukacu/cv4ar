#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

#include "marker.h"

#define WINDOW_NAME "Camera pose"

using namespace cv;
using namespace std;

typedef struct CameraPose {
    Mat transformation;
    unsigned int id;
} CameraPose;

Mat pattern, visualization;
float pattern_size = 50;
Size2f camera_size(8, 6);
vector<CameraPose> cameras;
Scalar camera_color(100, 255, 100);

bool process_image(Mat& image, Mat intrinsics, Mat distortion, Mat& rotation, Mat& translation) {

    image.copyTo(visualization);

    cvtColor(image, image, COLOR_BGR2GRAY);

    if (!detect_template_marker(image, pattern, pattern_size, intrinsics, distortion, rotation, translation))
        return false;


    Mat origin_points = (Mat_<float>(4,3) << 0, 0, 0, pattern_size, 0, 0, 0, pattern_size, 0, 0, 0, pattern_size);
    vector<Point3f> camera_points;
    camera_points.push_back(Point3f(0, 0, 0));
    camera_points.push_back(Point3f(-camera_size.width / 2, -camera_size.height / 2, 5));
    camera_points.push_back(Point3f(camera_size.width / 2, -camera_size.height / 2, 5));
    camera_points.push_back(Point3f(camera_size.width / 2, camera_size.height / 2, 5));
    camera_points.push_back(Point3f(-camera_size.width / 2, camera_size.height / 2, 5));

    std::vector<cv::Point3f> transformed_points;
    std::vector<cv::Point2f> projected_points;

    projectPoints(origin_points, rotation, translation, intrinsics, distortion, projected_points);

    line(visualization, projected_points.at(0), projected_points.at(1), Scalar(0,0,255), 3);
    line(visualization, projected_points.at(0), projected_points.at(2), Scalar(0,255,0), 3);
    line(visualization, projected_points.at(0), projected_points.at(3), Scalar(255,0,0), 3);

    for (unsigned int i = 0; i < cameras.size(); i++) {

        perspectiveTransform(camera_points, transformed_points, cameras[i].transformation);
        projectPoints(transformed_points, rotation, translation, intrinsics, distortion, projected_points);

        line(visualization, projected_points.at(0), projected_points.at(1), camera_color, 2);
        line(visualization, projected_points.at(0), projected_points.at(2), camera_color, 2);
        line(visualization, projected_points.at(0), projected_points.at(3), camera_color, 2);
        line(visualization, projected_points.at(0), projected_points.at(4), camera_color, 2);
        line(visualization, projected_points.at(1), projected_points.at(2), camera_color, 2);
        line(visualization, projected_points.at(2), projected_points.at(3), camera_color, 2);
        line(visualization, projected_points.at(3), projected_points.at(4), camera_color, 2);
        line(visualization, projected_points.at(4), projected_points.at(1), camera_color, 2);

        putText(visualization, format("%d", cameras[i].id), projected_points.at(0), FONT_HERSHEY_SIMPLEX, 1,
                camera_color, 1, 8);
    }

    return true;

}

int main(int argc, const char** argv) {

    string calibration_file("camera.yaml");
    string pattern_file("marker.png");
    int camera_id = 0;

    if (argc > 1) camera_id = atoi(argv[1]);
    if (argc > 2) pattern_file = argv[2];
    if (argc > 3) calibration_file = argv[3];

    pattern = imread(pattern_file, CV_LOAD_IMAGE_GRAYSCALE);

    VideoCapture camera(camera_id);

    if (!camera.isOpened()) {
        cout << "Unable to access camera" << endl;
        return -1;
    }

    if (pattern.empty()) {
        cout << "Pattern not found" << endl;
        return -1;
    }

    Mat frame, rectified, map1, map2, intrinsics, distortion;

    FileStorage fs(calibration_file, FileStorage::READ);

    fs["intrinsics"] >> intrinsics;
    fs["distortion"] >> distortion;

    Mat rotation, translation;

    cout << "Press space to capture new camera position and R to remove them all." << endl;

    while (true) {

        camera.read(frame);

        bool positioned = process_image(frame, intrinsics, distortion, rotation, translation);

        imshow(WINDOW_NAME, visualization);

        int c = waitKey(30);
        switch(c % 256) {
        case 'q':
        case 27: {
            return 0;
        }
        case 32: {
            if (positioned) {
                cout << "Creating camera" << endl;
                CameraPose pose;
                pose.id = (unsigned int) cameras.size();
                pose.transformation = Mat::eye(4, 4, CV_32FC1);
                Mat trotation;
                Rodrigues(rotation, trotation);
                trotation.copyTo(pose.transformation(Rect(0, 0, 3, 3)));
                translation.copyTo(pose.transformation(Rect(3, 0, 1, 3)));
                pose.transformation = pose.transformation.inv();
                cameras.push_back(pose);
            }
            break;
        }
        case 'c':
        case 'r': {
            cameras.clear();
            cout << "Removing all cameras" << endl;
            break;
        }
        }

    }

    return 0;
}
