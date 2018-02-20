#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//#define ADAPTIVE

#define WINDOW_NAME "Thresholding"

int threshold_value = 128;
int threshold_type = THRESH_BINARY;

#ifdef ADAPTIVE
int block_size = 15;
int threshold_adaptive = ADAPTIVE_THRESH_GAUSSIAN_C; //ADAPTIVE_THRESH_MEAN_C
int offset = 0;
#endif

int main( int argc, char** argv ) {

    Mat frame, src, dst;

    int camera_id = 0;
    if (argc > 1) camera_id = atoi(argv[1]);

    VideoCapture camera(camera_id);

    if (!camera.isOpened()) {
        cout << "Unable to access camera" << endl;
        return -1;
    }

    namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

    cout << "Press: \n * b: Binary \n * i: Binary Inverted \n * t: Truncate \n * z: To Zero \n * v: To Zero Inverted \n";

#ifndef ADAPTIVE
    createTrackbar("Threshold value", WINDOW_NAME, &threshold_value, 255);
#else
    createTrackbar("Block size", WINDOW_NAME, &block_size, 40);
    createTrackbar("Offset", WINDOW_NAME, &offset, 50);
#endif

    while (true) {

        camera.read(frame);

        cvtColor(frame, src, CV_BGR2GRAY);

#ifndef ADAPTIVE
        threshold(src, dst, threshold_value, 255, threshold_type);
#else
        adaptiveThreshold(src, dst, 255, threshold_adaptive, threshold_type, block_size * 2 + 1, offset - 25);
#endif

        imshow(WINDOW_NAME, dst);

        int c = waitKey(30);
        switch(c % 256) {
        case 'q':
        case 27: {
            return 0;
        }
        case 'b': {
            threshold_type = THRESH_BINARY;
            break;
        }
        case 'i': {
            threshold_type = THRESH_BINARY_INV;
            break;
        }
        case 't': {
            threshold_type = THRESH_TRUNC;
            break;
        }
        case 'z': {
            threshold_type = THRESH_TOZERO;
            break;
        }
        case 'v': {
            threshold_type = THRESH_TOZERO_INV;
            break;
        }
        }

    }

    return 0;

}


