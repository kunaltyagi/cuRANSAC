#ifndef CIRCLE_DETECT_HPP
#define CIRCLE_DETECT_HPP

#include <ransac_cuda.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/video/background_segm.hpp"


using namespace cv;

/**
 * @class CircleDetect
 * @brief Class implementing circle detection using
 *          RANSAC algorithm
 */
class CircleDetect
{
    public:
    /**
     * @brief Default ctor
     */
    CircleDetect();

    /**
     * @brief RANSAC based circle detection
     * @param[in] image Input image
     * @param[out] circles Detected circles
     * @param[in] canny_threshold Canny threshold
     * @param[in] numIterations Number of iterations of RANSAC
     */
    void RANSAC(Mat &image, std::vector<Vec3f> &circles, double canny_threshold, double circle_threshold, int numIterations);

    /**
     * @brief Groups concentric circles detected using RANSAC
     * @param[in] circles Detected circles
     * @return Vector of grouped circles
     */
    std::vector<std::vector<Vec3f> > group_concentric(std::vector<Vec3f> circles);
};

#endif  //CIRCLE_DETECT_HPP
