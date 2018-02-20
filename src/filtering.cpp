#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define WINDOW_NAME "Filtering"
#define TRACKBAR_NAME "Filter size"
#define MAX_KERNEL_SIZE 16

#define FILTER_BLUR 1
#define FILTER_GAUSSIAN_BLUR 2
#define FILTER_MEDIAN 3
#define FILTER_BILATERAL 4
#define FILTER_ERODE 5
#define FILTER_DILATE 6
#define FILTER_SOBEL_X 7
#define FILTER_SOBEL_Y 8
#define FILTER_SHARPEN 9

Mat src, dst, frame;
int filter_size = 1;
int filter_type = FILTER_BLUR;

void do_filtering() {

    int fsize = filter_size * 2 + 1;

    switch (filter_type) {
    case FILTER_BLUR: {
        blur(src, dst, Size(fsize, fsize), Point(-1, -1));
        break;
    }
    case FILTER_GAUSSIAN_BLUR: {
        GaussianBlur(src, dst, Size(fsize, fsize), 0, 0);
        break;
    }
    case FILTER_MEDIAN: {
        medianBlur (src, dst, fsize);
        break;
    }
    case FILTER_BILATERAL: {
        bilateralFilter (src, dst, fsize, fsize * 2, fsize / 2);
        break;
    }
    case FILTER_ERODE: {
        Mat element = getStructuringElement(MORPH_RECT,
                                            Size(fsize, fsize), Point(filter_size, filter_size));
        erode(src, dst, element);
        break;
    }
    case FILTER_DILATE: {
        Mat element = getStructuringElement(MORPH_RECT,
                                            Size(fsize, fsize), Point(filter_size, filter_size));
        dilate(src, dst, element);
        break;
    }
    case FILTER_SOBEL_X: {
        Sobel(src, dst, CV_16S, 1, 0, fsize);
        break;
    }
    case FILTER_SOBEL_Y: {
        Sobel(src, dst, CV_16S, 0, 1, fsize);
        break;
    }
    case FILTER_SHARPEN: {
        GaussianBlur(src, dst, Size(), fsize);
        addWeighted(src, 1.5, dst, -0.5, 0, dst);
        break;
    }
    }

}


int main( int argc, char** argv ) {

    int camera_id = 0;
    if (argc > 1) camera_id = atoi(argv[1]);

    VideoCapture camera(camera_id);

    if (!camera.isOpened()) {
        cout << "Unable to access camera" << endl;
        return -1;
    }

    namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

    cout << "Press: \n * b: Blur \n * g: Gaussian blur \n * m: Median \n * l: Bilateral \n * e: Erode \n * d: Dilate \n * x: Sobel X direction \n * y: Sobel Y direction \n * s: Sharpening \n";

    createTrackbar(TRACKBAR_NAME, WINDOW_NAME, &filter_size, MAX_KERNEL_SIZE);


    while (true) {

        camera.read(frame);

        cvtColor(frame, src, CV_BGR2GRAY);

        do_filtering();

        imshow(WINDOW_NAME, dst);

        int c = waitKey(30);
        switch(c % 256) {
        case 'q':
        case 27: {
            return 0;
        }
        case 'b': {
            filter_type = FILTER_BLUR;
            break;
        }
        case 'g': {
            filter_type = FILTER_GAUSSIAN_BLUR;
            break;
        }
        case 'm': {
            filter_type = FILTER_MEDIAN;
            break;
        }
        case 'l': {
            filter_type = FILTER_BILATERAL;
            break;
        }
        case 'e': {
            filter_type = FILTER_ERODE;
            break;
        }
        case 'd': {
            filter_type = FILTER_DILATE;
            break;
        }
        case 'x': {
            filter_type = FILTER_SOBEL_X;
            break;
        }
        case 'y': {
            filter_type = FILTER_SOBEL_Y;
            break;
        }
        case 's': {
            filter_type = FILTER_SHARPEN;
            break;
        }
        }
    }

    return 0;
}


