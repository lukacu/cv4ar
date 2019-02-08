
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "marker.h"

bool detect_template_marker_corners(Mat image, Mat marker, vector<Point2f>& image_points) {

    int offset = 5;
    int block_size = 45;
    double confidence_threshold = 0.8;
    double best_confidence = 0;
    Point2f region_corners[4];
    Mat temp, normalized(marker.size(), CV_8UC1);
    Point2f normalized_corners[4] = {Point2f(0, 0), Point2f((float)marker.rows - 1, 0),
                                     Point2f((float)marker.rows - 1, (float)marker.cols - 1), Point2f(0, (float)marker.cols - 1)
                                    };
    Mat rotated_marker[4];

    adaptiveThreshold(image, temp, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, block_size, offset);
    dilate(temp, temp, Mat());

    int avsize = (temp.rows + temp.cols) / 2;

    vector<vector<Point> > contours;
    vector<Point> polygon;

    findContours(temp, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    if(marker.cols != marker.rows)
        throw runtime_error("Not a square marker");

    unsigned int i;
    Point p;
    int pMinX, pMinY, pMaxY, pMaxX;

    marker.copyTo(rotated_marker[0]);

    for(i = 1; i < 4; i++) {
        Mat rotated;
        transpose(rotated_marker[i-1], rotated);
        flip(rotated, rotated_marker[i], 1);
    }

    for(i=0; i < contours.size(); i++) {

        Mat contour_matrix = Mat (contours[i]);
        const double per = arcLength(contour_matrix, true);

        if (per > (avsize / 4) && per < (4 * avsize)) {

            polygon.clear();
            approxPolyDP(contour_matrix, polygon, per * 0.02, true);

            //check rectangularity and convexity
            if (polygon.size() == 4 && isContourConvex(Mat(polygon))) {

                // determine the bounding box of contour
                p = polygon.at(0);
                pMinX = pMaxX = p.x;
                pMinY = pMaxY = p.y;
                int j;
                for(j=1; j<4; j++) {
                    p = polygon.at(j);
                    if (p.x < pMinX) pMinX = p.x;
                    if (p.x > pMaxX) pMaxX = p.x;
                    if (p.y < pMinY) pMinY = p.y;
                    if (p.y > pMaxY) pMaxY = p.y;
                }
                Rect bounding_box(pMinX, pMinY, pMaxX - pMinX + 1, pMaxY - pMinY + 1);

                //find the upper left vertex
                double dmin = (4 * avsize * avsize);
                int first_point = -1;
                for (j = 0; j < 4; j++) {
                    double d = norm(polygon.at(j));
                    if (d < dmin) {
                        dmin = d;
                        first_point = j;
                    }
                }

                vector<Point2f> refined(4);
                copy(polygon.begin(), polygon.end(), refined.begin());

                cornerSubPix(image, refined, Size(3,3), Size(-1,-1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

                for(j = 0; j < 4; j++)
                    region_corners[j] = Point2f(refined.at((4 + first_point - j) % 4).x - pMinX, refined.at((4 + first_point - j) % 4).y - pMinY);

                Mat H(3,3,CV_32F);
                H = getPerspectiveTransform(region_corners, normalized_corners);

                // warp the input based on the homography model to get the normalized interest region
                warpPerspective(image(Range(bounding_box.y, bounding_box.y + bounding_box.height),
                                      Range(bounding_box.x, bounding_box.x + bounding_box.width)),
                                normalized, H, Size(normalized.cols, normalized.rows));

                int orientation = 0;
                double confidence = 0;
                double N = (double)(marker.cols * marker.rows);

                Scalar mean_value, std_value;
                meanStdDev(normalized, mean_value, std_value);
                double normalized_squared = pow(norm(normalized), 2);

                for(j = 0; j < 4; j++) {

                    // NCC comparison with all four orientations
                    double const nnn = pow(norm(rotated_marker[j]), 2);
                    double const mmm = mean(rotated_marker[j]).val[0];
                    double nominator = normalized.dot(rotated_marker[j]) - (N * mean_value.val[0] * mmm);
                    double denominator = sqrt((normalized_squared - (N * mean_value.val[0] * mean_value.val[0]) ) * (nnn - (N * mmm * mmm)));
                    double temp_confidence = nominator / denominator;

                    if(temp_confidence > confidence) {
                        confidence = temp_confidence;
                        orientation = j;
                    }
                }

                if (confidence > best_confidence) {

                    best_confidence = confidence;

                    image_points.clear();

                    for (j=0; j<4; j++)
                        image_points.push_back(refined.at((8 - orientation + first_point - j) % 4));

                }
            }
        }
    }

    return best_confidence >= confidence_threshold;
}

bool detect_template_marker(Mat image, Mat pattern, float pattern_size, Mat intrinsics, Mat distortion, Mat& rotation, Mat& translation) {

    vector<Point3f> pattern_points;
    vector<Point2f> image_points;
    pattern_points.push_back(Point3f(0, 0, 0));
    pattern_points.push_back(Point3f(pattern_size, 0, 0));
    pattern_points.push_back(Point3f(pattern_size, pattern_size, 0));
    pattern_points.push_back(Point3f(0, pattern_size, 0));

    if (!detect_template_marker_corners(image, pattern, image_points)) return false;

    solvePnP(pattern_points, image_points, intrinsics, distortion, rotation, translation);

    return true;

}

bool detect_template_marker(Mat image, Mat pattern, float pattern_size, Mat& homography) {

    vector<Point2f> pattern_points;
    vector<Point2f> image_points;
    pattern_points.push_back(Point2f(0, 0));
    pattern_points.push_back(Point2f(pattern_size, 0));
    pattern_points.push_back(Point2f(pattern_size, pattern_size));
    pattern_points.push_back(Point2f(0, pattern_size));

    if (!detect_template_marker_corners(image, pattern, image_points)) return false;

    homography = findHomography(pattern_points, image_points);

    return true;

}



