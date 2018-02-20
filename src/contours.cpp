#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define WINDOW_NAME "Contours"

#define CONTOURS_ALL 0
#define CONTOURS_LARGE 1
#define CONTOURS_SMALL 2
#define CONTOURS_CONVEX 4
#define CONTOURS_RECTANGLE 8

Mat src, dst, frame;

int mode;

int main( int argc, char** argv ) {

    int camera_id = 0;
    if (argc > 1) camera_id = atoi(argv[1]);

    VideoCapture camera(camera_id);

    if (!camera.isOpened()) {
        cout << "Unable to access camera" << endl;
        return -1;
    }

    namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

    cout << "Press: \n * a: All \n * f: Filtered \n";

    while (true) {

        camera.read(frame);

        cvtColor(frame, src, CV_BGR2GRAY);

        int offset = 5;
        int block_size = 45;

        adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, block_size, offset);
        dilate(dst, dst, Mat());

        vector<vector<Point> > contours;
        vector<Point> polygon;

        findContours(dst, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        if (mode == CONTOURS_ALL) {
            drawContours(frame, contours, -1, Scalar(255, 255, 0));
        } else {

            int image_perimiter = (dst.rows + dst.cols) / 2;

            for(size_t i = 0; i < contours.size(); i++) {

                Mat contour_matrix = Mat(contours[i]);
                double perimiter = arcLength(contour_matrix, true);

                if (perimiter > (image_perimiter / 4) || !(mode & CONTOURS_LARGE)) {
                    if (perimiter < (4 * image_perimiter) || !(mode & CONTOURS_SMALL)) {

                        polygon.clear();
                        approxPolyDP(contour_matrix, polygon, perimiter * 0.02, true);

                        //check rectangularity and convexity
                        if (polygon.size() == 4 || !(mode & CONTOURS_RECTANGLE))
                            if (isContourConvex(Mat(polygon)) || !(mode & CONTOURS_CONVEX)) {
                                drawContours(frame, contours, (int)i, Scalar(255, 255, 0));
                            }
                    }
                }
            }
        }

        imshow(WINDOW_NAME, frame);

        int c = waitKey(30);
        switch(c % 256) {
        case 'q':
        case 27: {
            return 0;
        }
        case 'a': {
            mode = CONTOURS_ALL;
            break;
        }
        case 'l': {
            mode ^= CONTOURS_LARGE;
            break;
        }
        case 'h': {
            mode ^= CONTOURS_SMALL;
            break;
        }
        case 'c': {
            mode ^= CONTOURS_CONVEX;
            break;
        }
        case 's': {
            mode ^= CONTOURS_RECTANGLE;
            break;
        }
        }

    }

    return 0;
}


