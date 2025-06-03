#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <queue>

using namespace std;
using namespace cv;

Mat applyMedianFilter(const Mat& src, int ksize = 3) {
    int half = ksize / 2;
    Mat dst = Mat::zeros(src.size(), src.type());

    Mat padded;
    copyMakeBorder(src, padded, half, half, half, half, BORDER_REFLECT101);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            vector<uchar> window;

            for (int i = -half; i <= half; ++i) {
                for (int j = -half; j <= half; ++j) {
                    window.push_back(padded.at<uchar>(y + i + half, x + j + half));
                }
            }

            sort(window.begin(), window.end());
            dst.at<uchar>(y, x) = window[window.size() / 2];
        }
    }

    return dst;
}


Mat createGaussianKernel(int ksize, double sigma) {
    int half = ksize / 2;
    Mat kernel(ksize, ksize, CV_64F);
    double sum = 0.0;
    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            double value = (1.0 / (2 * CV_PI * sigma * sigma)) *
                exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel.at<double>(i + half, j + half) = value;
            sum += value;
        }
    }
    kernel /= sum;
    return kernel;
}

Mat applyGaussianFilter(const Mat& src, const Mat& kernel) {
    int ksize = kernel.rows;
    int half = ksize / 2;
    Mat dst = Mat::zeros(src.size(), src.type());
    Mat padded;
    copyMakeBorder(src, padded, half, half, half, half, BORDER_REFLECT101);
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            vector<uchar> window;

            for (int i = 0; i < ksize; ++i) {
                for (int j = 0; j < ksize; ++j) {
                    window.push_back(padded.at<uchar>(y + i, x + j));
                }
            }

            sort(window.begin(), window.end());
            dst.at<uchar>(y, x) = window[window.size() / 2];
        }
    }

    return dst;
}

void computeGradientAndDirection(const Mat& image, Mat& magnitude, Mat& direction) {
    int kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int ky[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    magnitude = Mat::zeros(image.size(), CV_64F);
    direction = Mat::zeros(image.size(), CV_64F);
    Mat padded;
    copyMakeBorder(image, padded, 1, 1, 1, 1, BORDER_REFLECT101);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            double gx = 0.0, gy = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    uchar pixel = padded.at<uchar>(y + i, x + j);
                    gx += pixel * kx[i][j];
                    gy += pixel * ky[i][j];
                }
            }
            magnitude.at<double>(y, x) = sqrt(gx * gx + gy * gy);
            direction.at<double>(y, x) = atan2(gy, gx) * 180.0 / CV_PI;
            if (direction.at<double>(y, x) < 0)
                direction.at<double>(y, x) += 180;
        }
    }
}

Mat nonMaximumSuppression(const Mat& magnitude, const Mat& direction) {
    Mat suppressed = Mat::zeros(magnitude.size(), CV_64F);
    for (int y = 1; y < magnitude.rows - 1; ++y) {
        for (int x = 1; x < magnitude.cols - 1; ++x) {
            double angle = direction.at<double>(y, x);
            double mag = magnitude.at<double>(y, x);
            double q = 0.0, r = 0.0;
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                q = magnitude.at<double>(y, x + 1);
                r = magnitude.at<double>(y, x - 1);
            } else if (angle >= 22.5 && angle < 67.5) {
                q = magnitude.at<double>(y - 1, x + 1);
                r = magnitude.at<double>(y + 1, x - 1);
            } else if (angle >= 67.5 && angle < 112.5) {
                q = magnitude.at<double>(y - 1, x);
                r = magnitude.at<double>(y + 1, x);
            } else if (angle >= 112.5 && angle < 157.5) {
                q = magnitude.at<double>(y + 1, x + 1);
                r = magnitude.at<double>(y - 1, x - 1);
            }
            if (mag >= q && mag >= r)
                suppressed.at<double>(y, x) = mag;
        }
    }
    return suppressed;
}

Mat cannyEdgeDetection(const Mat& image) {
    //Mat denoised = applyMedianFilter(image, 5);
    Mat kernel = createGaussianKernel(7, 2.5);
    Mat blurred = applyGaussianFilter(image, kernel);
    Mat magnitude, direction;
    computeGradientAndDirection(blurred, magnitude, direction);
    Mat suppressed = nonMaximumSuppression(magnitude, direction);

    double minVal, maxVal;
    minMaxLoc(suppressed, &minVal, &maxVal);
    double highThreshold = 0.3 * maxVal;
    double lowThreshold = 0.1 * maxVal;

    Mat edges = Mat::zeros(suppressed.size(), CV_8U);
    for (int i = 1; i < suppressed.rows - 1; ++i) {
        for (int j = 1; j < suppressed.cols - 1; ++j) {
            double val = suppressed.at<double>(i, j);
            if (val >= highThreshold) {
                edges.at<uchar>(i, j) = 255;
            } else if (val >= lowThreshold) {
                edges.at<uchar>(i, j) = 128;
            }
        }
    }

    queue<Point> q;
    for (int y = 1; y < edges.rows - 1; y++) {
        for (int x = 1; x < edges.cols - 1; x++) {
            if (edges.at<uchar>(y, x) == 255) {
                q.push(Point(x, y));
            }
        }
    }

    while (!q.empty()) {
        Point p = q.front(); q.pop();
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = p.x + dx;
                int ny = p.y + dy;
                if (edges.at<uchar>(ny, nx) == 128) {
                    edges.at<uchar>(ny, nx) = 255;
                    q.push(Point(nx, ny));
                }
            }
        }
    }

    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            if (edges.at<uchar>(y, x) == 128)
                edges.at<uchar>(y, x) = 0;
        }
    }

    return edges;
}

void houghTransform(const Mat& edgeImage, vector<pair<int, int>>& lines, int threshold = 60) {
    int width = edgeImage.cols;
    int height = edgeImage.rows;
    int numAngles = 180;
    vector<double> sinTable(numAngles), cosTable(numAngles);
    for (int t = 0; t < numAngles; ++t) {
        double theta = (t - 90) * CV_PI / 180.0;
        sinTable[t] = sin(theta);
        cosTable[t] = cos(theta);
    }
    int maxRho = static_cast<int>(sqrt(width * width + height * height));
    int accumulatorHeight = 2 * maxRho;
    Mat accumulator = Mat::zeros(accumulatorHeight, numAngles, CV_32SC1);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (edgeImage.at<uchar>(y, x) > 250) {
                for (int t = 0; t < numAngles; ++t) {
                    double rho = x * cosTable[t] + y * sinTable[t];
                    int r = static_cast<int>(round(rho)) + maxRho;
                    if (r >= 0 && r < accumulatorHeight) {
                        accumulator.at<int>(r, t)++;
                    }
                }
            }
        }
    }

    for (int r = 0; r < accumulator.rows; ++r) {
        for (int t = 0; t < accumulator.cols; ++t) {
            if (accumulator.at<int>(r, t) >= threshold) {
                lines.emplace_back(r - maxRho, t - 90);
            }
        }
    }
}

void drawHoughLines(Mat& image, const vector<pair<int, int>>& lines) {
    for (const auto& houghLine : lines) {
        int rho = houghLine.first;
        double thetaDeg = houghLine.second;
        if (thetaDeg < 20 || thetaDeg > 160) continue;

        double theta = thetaDeg * CV_PI / 180.0;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

        if (norm(pt1 - pt2) > 100)
            line(image, pt1, pt2, Scalar(0, 255, 0), 2, LINE_AA);
    }
}

vector<pair<int, int>> filterSimilarLines(const vector<pair<int, int>>& lines, int rhoThresh = 10, int thetaThresh = 1) {
    vector<pair<int, int>> filtered;
    for (const auto& current : lines) {
        bool isDuplicate = false;
        for (const auto& existing : filtered) {
            if (abs(current.first - existing.first) < rhoThresh &&
                abs(current.second - existing.second) < thetaThresh) {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) {
            filtered.push_back(current);
        }
    }
    return filtered;
}

int main() {
    Mat image = imread("../resources/horizontal-lines.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Eroare: Imaginea nu a putut fi încărcată.\n";
        return 1;
    }

    Mat edges = cannyEdgeDetection(image);

    vector<pair<int, int>> lines;
    houghTransform(edges, lines, 60);
    lines = filterSimilarLines(lines, 20, 2);

    Mat colorImage;
    cvtColor(edges, colorImage, COLOR_GRAY2BGR);
    drawHoughLines(colorImage, lines);

    imshow("Imagine originala grayscale", image);
    imshow("Muchii (Canny adaptiv)", edges);
    imshow("Linii detectate (Hough filtrat)", colorImage);

    waitKey(0);
    return 0;
}
