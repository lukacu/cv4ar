#include <cstdio>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#ifdef _XFEATURES
#include <opencv2/xfeatures2d.hpp>
using namespace cv::xfeatures2d;
#endif

#define WINDOW_NAME "Correspondences"

Ptr<DescriptorMatcher> matcher = new BFMatcher();
Ptr<Feature2D> descriptor;

Mat frame, gray, video_frame, pattern, visualization;

Mat descriptors_pattern;
vector<KeyPoint> keypoints_pattern;

Ptr<Feature2D> get_descriptor(int id = 'a') {
    Ptr<Feature2D> d;

    switch (id) {
#ifdef _XFEATURES
    case 's': {
        d = SIFT::create();
        matcher = new BFMatcher();
        break;
    }
    case 'u': {
        d = SURF::create();
        matcher = new BFMatcher();
        break;
    }
#endif
    case 'o': {
        d = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
        matcher = new BFMatcher(NORM_HAMMING);
        break;
    }
    case 'a': {
        d = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 4, 4, KAZE::DIFF_PM_G2);
        matcher = new BFMatcher(NORM_HAMMING);
        break;
    }
    }

    return d;
}

bool match_comparator (DMatch& i, DMatch& j) {
    return (i.distance<j.distance);
}


int main( int argc, char** argv ) {

    string pattern_file("pattern.jpg");
    int camera_id = 0;

    if (argc > 1) camera_id = atoi(argv[1]);
    if (argc > 2) pattern_file = argv[2];

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

    namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

#ifdef _XFEATURES
    cout << "Press: \n * s: SIFT \n * u: SURF \n * o: ORB \n * a: AKAZE \n";
#else
    cout << "Press: \n * o: ORB \n * a: AKAZE \n";
#endif

    descriptor = get_descriptor();
    descriptor->detectAndCompute(pattern, Mat(), keypoints_pattern, descriptors_pattern);

    int filter = 10;
    createTrackbar("Filter matches", WINDOW_NAME, &filter, 100);

    while (true) {

        camera.read(frame);

        if (frame.empty())
            break;

        cvtColor(frame, gray, CV_BGR2GRAY);

        vector<KeyPoint> keypoints_gray;
        Mat descriptors_gray;
        vector<DMatch> matches;

        descriptor->detectAndCompute(gray, Mat(), keypoints_gray, descriptors_gray);
        matcher->match(descriptors_pattern, descriptors_gray, matches);

        std::sort (matches.begin(), matches.end(), match_comparator);

        std::vector<DMatch> matches_filter(matches.begin(), matches.begin() + (matches.size() * filter) / 100);

        drawMatches(pattern, keypoints_pattern, gray, keypoints_gray, matches_filter, visualization);
        imshow(WINDOW_NAME, visualization);

        int c = waitKey(30);
        switch(c % 256) {
        case -1:
            break;
        case 'q':
        case 27: {
            return 0;
        }
        default: {
            descriptor = get_descriptor(c % 256);
            keypoints_pattern.clear();
            descriptors_pattern.release();
            descriptor->detectAndCompute(pattern, Mat(), keypoints_pattern, descriptors_pattern);

            break;
        }
        }

    }

    return 0;

}


