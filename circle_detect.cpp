#include <circle_detect.hpp>

#define DEBUG 1

using namespace cv;

CircleDetect::CircleDetect()
{}

void CircleDetect::RANSAC(Mat &image, std::vector<Vec3f> &circles, double cannyThreshold,
                            double circleThreshold, int numIterations)
{
    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);
    circles.clear();

    Mat edges;
    Canny(image, edges, MAX(cannyThreshold/2,1), cannyThreshold, 3);
    std::vector<std::vector<Point > > contours;
    imshow("Edges", edges);    
    //waitKey(0);
    findContours(edges,contours,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);

#if DEBUG == 1
    char* source_window = "Source";
    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    imshow( source_window, image );
    source_window = "Canny";
    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    imshow( source_window, edges );
//..    waitKey(0);
    /*
    RNG rng(12345);
    Mat drawing;// = Mat::zeros( contours.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
     {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color);//, 2, 8, hierarchy, 0, Point() );
     }
     namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
     imshow( "Contours", drawing );
     */
#endif

    ransac_common_params ransacParams;
    ransacParams.max_point_separation = (int)image.cols;
    std::cout << "CIRCLE DETECT : " << ransacParams.max_point_separation << std::endl;
    ransacParams.min_point_separation = (int)image.cols/50;
    ransacParams.colinear_tolerance = 1;
    ransacParams.radius_tolerance = (int)image.cols/2;
    ransacParams.points_threshold = 10;
    ransacParams.max_radius = (int)image.cols/4;
    ransacParams.circle_threshold = circleThreshold;
    
    int i,j;
    if (contours.size() < 200 && contours.size() > 0)
    {
        ransac_result* ransacResult = new ransac_result [contours.size()];
        ransac_contour_params* contourParams = new ransac_contour_params [contours.size()];
        ransacParams.num_iterations = (int)numIterations;///contours.size();
        ransacParams.num_contours = contours.size();

        int** consensus_x = new int* [contours.size()];
        int** consensus_y = new int* [contours.size()];
        for(i=0; i<contours.size(); i++)
        {
            consensus_x[i] = new int [contours[i].size()];
            consensus_y[i] = new int [contours[i].size()];
            for(j=0; j<contours[i].size(); j++)
            {
                consensus_x[i][j] = contours[i][j].x;
                consensus_y[i][j] = contours[i][j].y;
            }
            contourParams[i].consensus_size = contours[i].size();
            contourParams[i].sample_size = 3;
        }
        launch_ransac_kernels(consensus_x, consensus_y, &ransacParams,
                                contourParams, ransacResult);

        circle(image, Point(ransacResult[0].cx,ransacResult[0].cy), ransacResult[0].radius, 130);
        imshow("Result", image);
        waitKey(0);

        delete[] ransacResult;
        delete[] contourParams;
        for(i=0; i<contours.size(); i++)
        {
            delete[] consensus_x[i];
            delete[] consensus_y[i];
        }
        delete[] consensus_x;
        delete[] consensus_y;
    }
}
