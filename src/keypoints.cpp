#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#define WINDOW_NAME "Features"

#ifdef _XFEATURES
#include <opencv2/xfeatures2d.hpp>
using namespace cv::xfeatures2d;
#endif

Ptr<FeatureDetector> detector;

Ptr<FeatureDetector> get_detector(int id = 'a') {
    Ptr<Feature2D> d;

    switch (id) {
#ifdef _XFEATURES
    case 's': {
        d = SIFT::create();
        break;
    }
    case 'u': {
        d = SURF::create(400);
        break;
    }
#endif
    case 'f': {
        d = FastFeatureDetector::create();
        break;
    }
    case 'b': {
        d = BRISK::create();
        break;
    }
    case 'o': {
        d = ORB::create();
        break;
    }
    case 'm': {
        d = MSER::create();
        break;
    }
    default: {
        d = AKAZE::create();
        break;
    }
    }

    return d;
}


void do_detect(Mat img) {

    Mat visualization;
    std::vector<KeyPoint> keypoints;

    detector->detect(img, keypoints);

    drawKeypoints(img, keypoints, visualization, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imshow(WINDOW_NAME, visualization);

}

int main( int argc, char** argv ) {

    Mat frame, gray, video_frame, marker;

    int camera_id = 0;

    if (argc > 1) camera_id = atoi(argv[1]);

    VideoCapture camera(camera_id);

    if (!camera.isOpened()) {
        cout << "Unable to access camera" << endl;
        return -1;
    }

    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);

#ifdef _XFEATURES
    cout << "Press: \n * s: SIFT \n * u: SURF \n * f: FAST \n * b: BRISK \n * o: ORB \n * m: MSER \n * a: AKAZE \n";
#else
    cout << "Press: \n * f: FAST \n * b: BRISK \n * o: ORB \n * m: MSER \n * a: AKAZE \n";
#endif

    detector = get_detector();

    while (true) {

        camera.read(frame);

        if (frame.empty())
            break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        do_detect(gray);

        int c = waitKey(30);
        switch(c % 256) {
        case -1:
            break;
        case 'q':
        case 27: {
            return 0;
        }
        default: {
            detector = get_detector(c % 256);
            break;
        }
        }
    }

    return 0;
}


