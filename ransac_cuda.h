/**
  * Assuming Number of iterations is less than Threads per Block
  */

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "curand_kernel.h"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) {  printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)
//#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \ printf("Error at %s:%d\n",__FILE__,__LINE__);\ return EXIT_FAILURE;}} while(0)

struct ransac_common_params {
    float max_point_separation;
    float min_point_separation;
    float colinear_tolerance;
    float radius_tolerance;
    float points_threshold;
    float max_radius;
    float circle_threshold;
    int num_iterations;
    int num_contours;
};

struct ransac_contour_params {
    int consensus_size, sample_size;
};

struct ransac_result {
    int cx, cy;
    float radius;
};

void launch_ransac_kernels(int** points_x, int** points_y, ransac_common_params * params,
                            ransac_contour_params * contour_params, ransac_result * result);
