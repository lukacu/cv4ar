#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define WINDOW_NAME "Edges"
#define TRACKBAR_NAME_LOW "Hysteresis low"
#define TRACKBAR_NAME_HIGH "Hysteresis high"
#define TRACKBAR_NAME_KERNEL "Smoothing"
#define MAX_THRESHOLD_SIZE 1000
#define MAX_KERNEL_SIZE 3

Mat src, dst, frame;

int hysteresis_low = 5;
int hysteresis_high = 50;
int kernel_size = 1;

void do_canny() {

    Canny(src, dst, hysteresis_low, hysteresis_high, MAX(1, kernel_size) * 2 + 1);

    imshow(WINDOW_NAME, dst);
}


int main(int argc, char** argv) {

    int camera_id = 0;
    if (argc > 1) camera_id = atoi(argv[1]);

    VideoCapture camera(camera_id);

    if (!camera.isOpened()) {
        cout << "Unable to access camera" << endl;
        return -1;
    }

    src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

    createTrackbar(TRACKBAR_NAME_LOW, WINDOW_NAME, &hysteresis_low, MAX_THRESHOLD_SIZE);
    createTrackbar(TRACKBAR_NAME_HIGH, WINDOW_NAME, &hysteresis_high, MAX_THRESHOLD_SIZE);
    createTrackbar(TRACKBAR_NAME_KERNEL, WINDOW_NAME, &kernel_size, MAX_KERNEL_SIZE);

    while (true) {

        camera.read(frame);

        cvtColor(frame, src, CV_BGR2GRAY);

        do_canny();

        imshow(WINDOW_NAME, dst);

        if (waitKey(30) > 0) break;

    }

    return 0;

}
