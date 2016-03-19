#include <iostream>
#include <circle_detect.hpp>

int main (int argc, char* argv[])
{
    /*
    try
    {
        cv::Mat src_host = cv::imread("file.png", CV_LOAD_IMAGE_GRAYSCALE);
        cv::gpu::GpuMat dst, src;
        src.upload(src_host);

        cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

        //cv::Mat result_host = dst;
        cv::Mat result_host;
        dst.download(result_host);
        cv::imshow("Result", result_host);
        cv::waitKey();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    */

    Mat src;
    std::vector<Vec3f> circles;
    if (argc == 2)
    src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    else
    src = imread("test_images/circle.png", CV_LOAD_IMAGE_GRAYSCALE );
    CircleDetect cd;
    cd.RANSAC(src, circles, 100, 2, 30);
    return 0;
}

