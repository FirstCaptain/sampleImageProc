#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
// #include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;

void print_usage()
{
    std::cout << "./sample <filename>" << std::endl;
}

int main(int argc, const char *argv[])
{
    // simple argument parsing
    if (argc < 2)
    {
        std::cerr << "Filepath not provided" << std::endl;
        print_usage();
        return 1;
    }
    std::string image_path = argv[1];
    std::cout << "Filepath: " << image_path << std::endl;
    if (!std::filesystem::exists(image_path))
    {
        std::cerr << "Filepath does not exist" << std::endl;
        print_usage();
        return 1;
    }
    // read in a frame... only tested this with the image included as part of this repo. 
    // other factors/ domain knowledge would need to be considered to segment out an object from different scenes
    // this is a simple color segmentation. 
    Mat img;
    Mat frame = imread(image_path, IMREAD_COLOR);
    Mat orig = frame.clone();
    GaussianBlur(frame, frame, Size(5, 5), 0); // blur it a bit
    inRange(frame, Scalar(150, 150, 150), Scalar(255, 255, 255), img);

    Mat element = getStructuringElement(MORPH_RECT,
                                        Size(5, 5),
                                        Point(0, 0));
    // erode
    erode(img, img, element);
    erode(img, img, element);

    // dilate
    dilate(img, img, element);
    dilate(img, img, element);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    bitwise_not(img, img);
    // find the contours to set as mask
    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    drawContours(frame, contours, -1, Scalar(0, 255, 0), 2);

    // create a mask matching the outside of the contours
    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    cv::drawContours(mask, contours, -1, Scalar(255, 255, 255), cv::FILLED);
    imshow("mask", mask);

    // use laplacian to get a measure of focus with the masked region
    Mat lapl_img;
    Laplacian(frame, lapl_img, CV_64F);
    Scalar mean, stddev;
    meanStdDev(lapl_img, mean, stddev, img);
    double variance = stddev.val[0] * stddev.val[0];
    std::cout << "Threshold Focus Score: " << variance << std::endl;
    putText(frame, "Focus " + to_string(variance), Point(50, 50), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 0, 0));
    imshow("Output", frame);
    waitKey(0);
    imwrite("../output.png", frame);
    return 0;
}