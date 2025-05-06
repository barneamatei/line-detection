#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>


using namespace std;
using namespace cv;

Mat createGaussianKernel(int ksize, double sigma) {
    int half = ksize / 2;
    Mat kernel(ksize, ksize, CV_64F);
    double sum = 0.0;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            double value = (1.0 / (2 * CV_PI * sigma * sigma)) *
                std::exp(-(i * i + j * j) / (2 * sigma * sigma));
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

    for (int y = half; y < src.rows - half; ++y) {
        for (int x = half; x < src.cols - half; ++x) {
            double sum = 0.0;
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    uchar pixel = src.at<uchar>(y + ky, x + kx);
                    double weight = kernel.at<double>(ky + half, kx + half);
                    sum += pixel * weight;
                }
            }
            dst.at<uchar>(y, x) = static_cast<uchar>(sum);
        }
    }

    return dst;
}

Mat computeGradient(const Mat& image) {
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

    Mat magnitude = Mat::zeros(image.size(), CV_64F);

    for (int y = 1; y < image.rows - 1; ++y) {
        for (int x = 1; x < image.cols - 1; ++x) {
            double gx = 0.0, gy = 0.0;

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    uchar pixel = image.at<uchar>(y + i, x + j);
                    gx += pixel * kx[i + 1][j + 1];
                    gy += pixel * ky[i + 1][j + 1];
                }
            }

            magnitude.at<double>(y, x) = std::sqrt(gx * gx + gy * gy);
        }
    }

    return magnitude;
}


Mat cannyEdgeDetection(const Mat& image, double lowThreshold, double highThreshold) {
    Mat kernel = createGaussianKernel(5, 1.0);
    Mat blurred = applyGaussianFilter(image, kernel);

    Mat magnitude = computeGradient(blurred);

    Mat edges = Mat::zeros(magnitude.size(), CV_8U);
    for (int i = 0; i < magnitude.rows; ++i) {
        for (int j = 0; j < magnitude.cols; ++j) {
            if (magnitude.at<double>(i, j) > highThreshold) {
                edges.at<uchar>(i, j) = 255;
            } else if (magnitude.at<double>(i, j) > lowThreshold) {
                edges.at<uchar>(i, j) = 127;
            }
        }
    }

    return edges;
}


void houghTransform(const Mat& edgeImage, vector<pair<int, int>>& lines, int threshold = 100) {
    int width = edgeImage.cols;
    int height = edgeImage.rows;

    int numAngles = 180;
    vector<double> sinTable(numAngles);
    vector<double> cosTable(numAngles);
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
                lines.emplace_back(r - maxRho, t - 90);  // ρ, θ
            }
        }
    }
}

void drawHoughLines(Mat& image, const vector<pair<int, int>>& lines) {
    for (const auto& houghLine : lines) {
        int rho = houghLine.first;
        double theta = (houghLine.second) * CV_PI / 180.0;

        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;

        Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

        line(image, pt1, pt2, Scalar(0, 0, 255), 1);
    }
}

vector<pair<int, int>> filterSimilarLines(const vector<pair<int, int>>& lines, int rhoThresh = 15, int thetaThresh = 2) {
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
    Mat image = imread("../resources/TestImage1.jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        cerr << "Eroare: Imaginea nu a putut fi încărcată.\n";
        return 1;
    }

    Mat kernel = createGaussianKernel(7, 2.5);
    Mat blurred = applyGaussianFilter(image, kernel);

    double lowThreshold = 30;
    double highThreshold = 100;
    Mat edges = cannyEdgeDetection(blurred, lowThreshold, highThreshold);

    vector<pair<int, int>> lines;
    houghTransform(edges, lines, 180);

    lines = filterSimilarLines(lines);

    Mat colorImage;
    cvtColor(edges, colorImage, COLOR_GRAY2BGR);
    drawHoughLines(colorImage, lines);

    imshow("Imagine originala grayscale", image);
    imshow("Dupa filtrul Gaussian", blurred);
    imshow("Muchii (Canny)", edges);
    imshow("Linii detectate (Hough)", colorImage);

    waitKey(0);
    return 0;
}
