#include <ransac_cuda.h>
__global__
void ransac_kernel(int** consensus_set_x, int** consensus_set_y,
                    ransac_common_params* common_params,
                    ransac_contour_params* contour_params,
                    ransac_result* result)
{
    int contour_id = blockIdx.x;
    curandStateMRG32k3a state;
    curand_init(clock(), threadIdx.x, 0, &state);
    int consensus_size = contour_params->consensus_size;
    __shared__ int* votes;
    votes = (int*)(malloc(sizeof(int)*common_params->num_iterations));
    __shared__ ransac_result* iter_result;
    iter_result = (ransac_result*)(malloc(sizeof(ransac_result)*common_params->num_iterations));

    iter_result[threadIdx.x].radius = 0;
    iter_result[threadIdx.x].cy = 0;
    iter_result[threadIdx.x].cx = 0;
    if(threadIdx.x == 0)
    {
        result[contour_id].radius = 0;
        result[contour_id].cx = 0;
        result[contour_id].cy = 0;
    }

    unsigned int iA = curand(&state);
    unsigned int iB = curand(&state);
    unsigned int iC = curand(&state);
    iA %= consensus_size;
    iB %= consensus_size;
    iC %= consensus_size;
    float AB, BC, CA;
    float m_AB, m_BC, b_AB;//, b_BC;
    float x_mp_AB, y_mp_AB, x_mp_BC, y_mp_BC;
    float m_pb_AB, m_pb_BC, b_pb_AB, b_pb_BC;
#if DEBUG == 1
    if(threadIdx.x == 0 )
    printf("A B C : %d %d %d\n", iA, iB, iC);
#endif
    
    AB = norm3df(consensus_set_x[contour_id][iA] - consensus_set_x[contour_id][iB],
                 consensus_set_y[contour_id][iA] - consensus_set_y[contour_id][iB], 0.0);
    BC = norm3df(consensus_set_x[contour_id][iB] - consensus_set_x[contour_id][iC],
                 consensus_set_y[contour_id][iB] - consensus_set_y[contour_id][iC], 0.0);
    CA = norm3df(consensus_set_x[contour_id][iC] - consensus_set_x[contour_id][iA],
                 consensus_set_y[contour_id][iC] - consensus_set_y[contour_id][iA], 0.0);
#if DEBUG == 1
    if(threadIdx.x == 0 && blockIdx.x ==0)
    {
        printf("\nA : %d %d\n", consensus_set_x[contour_id][iA], consensus_set_y[contour_id][iA] );
        printf("B : %d %d\n", consensus_set_x[contour_id][iB], consensus_set_y[contour_id][iB] );
        printf("C : %d %d\n", consensus_set_x[contour_id][iC], consensus_set_y[contour_id][iC] );
        printf("AB BC : %f %f\n", AB, BC );
    }
#endif
    if(AB < common_params->min_point_separation || BC < common_params->min_point_separation ||
       CA < common_params->min_point_separation ||
       AB > common_params->max_point_separation || BC > common_params->max_point_separation ||
       CA > common_params->max_point_separation )
    {
#if DEBUG == 1
        if(threadIdx.x == 0 && blockIdx.x == 0)
        printf("\n\tAB, CA, Max pt separation %f %f %f\n", AB, CA, common_params->max_point_separation);// (float)(consensus_set_x[contour_id][iA] - consensus_set_x[contour_id][iB]), consensus_set_x[contour_id][iC] - consensus_set_x[contour_id][iA], common_params->max_point_separation);
#endif
        return ;
    }

    m_AB = (consensus_set_y[contour_id][iA]-consensus_set_y[contour_id][iB])/(consensus_set_x[contour_id][iA]-consensus_set_x[contour_id][iB] + 0.001);
    m_BC = (consensus_set_y[contour_id][iB]-consensus_set_y[contour_id][iC])/(consensus_set_x[contour_id][iB]-consensus_set_x[contour_id][iC] + 0.001);
    b_AB = (consensus_set_y[contour_id][iB] - m_AB*consensus_set_x[contour_id][iB]);

    if(abs(consensus_set_y[contour_id][iC] - ((m_AB*(consensus_set_x[contour_id][iC])) + b_AB )) < common_params->colinear_tolerance)
    {
        return ;
    }

    x_mp_AB = (consensus_set_x[contour_id][iA]+consensus_set_x[contour_id][iB])/2.0;
    y_mp_AB = (consensus_set_y[contour_id][iA]+consensus_set_y[contour_id][iB])/2.0;
    x_mp_BC = (consensus_set_x[contour_id][iB]+consensus_set_x[contour_id][iC])/2.0;
    y_mp_BC = (consensus_set_y[contour_id][iB]+consensus_set_y[contour_id][iC])/2.0;

    m_pb_AB = -1/m_AB;
    m_pb_BC = -1/m_BC;
    b_pb_AB = y_mp_AB - m_pb_AB*x_mp_AB;
    b_pb_BC = y_mp_BC - m_pb_BC*x_mp_BC;
    
    iter_result[threadIdx.x].cx = (b_pb_AB - b_pb_BC)/(m_pb_BC - m_pb_AB + 0.0001);
    iter_result[threadIdx.x].cy = m_pb_AB*iter_result[threadIdx.x].cx + b_pb_AB;
    iter_result[threadIdx.x].radius = norm3df(iter_result[threadIdx.x].cx - consensus_set_x[contour_id][iA],
                                iter_result[threadIdx.x].cx - consensus_set_y[contour_id][iA], 0.0);
    if(iter_result[threadIdx.x].cx < 0 || iter_result[threadIdx.x].cy < 0)
    {
        printf("AYYOOOOO\n");
        printf("In Thread %d %d %d %f\n", threadIdx.x, iter_result[threadIdx.x].cx,iter_result[threadIdx.x].cy,iter_result[threadIdx.x].radius);
        printf("\nA : %d %d\n", consensus_set_x[contour_id][iA], consensus_set_y[contour_id][iA] );
        printf("B : %d %d\n", consensus_set_x[contour_id][iB], consensus_set_y[contour_id][iB] );
        printf("C : %d %d\n", consensus_set_x[contour_id][iC], consensus_set_y[contour_id][iC] );
        printf("AB BC : %f %f\n", AB, BC );
    }

    votes[threadIdx.x] = 0;
    for (int i = 0; i < contour_params[contour_id].consensus_size; i++)
    {
        if (norm3df(consensus_set_y[contour_id][i] - iter_result[threadIdx.x].cy,
                    consensus_set_x[contour_id][i] - iter_result[threadIdx.x].cx, 0.0) - iter_result[threadIdx.x].radius < common_params->radius_tolerance)
        {
            votes[threadIdx.x]++;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        int max_votes = 0;
        int max_iter = -1;
        for (int i = 0; i < common_params->num_iterations; i++)
        {
            if (votes[i] > max_votes)
            {
                max_votes = votes[i];
                max_iter = i;
            }
        }
            printf("MAX ITER third %d",  common_params->num_iterations);
        if (max_votes > common_params->points_threshold && max_iter != -1)
        {
            result[contour_id].cx = iter_result[max_iter].cx;
            result[contour_id].cy = iter_result[max_iter].cy;
            result[contour_id].radius = iter_result[max_iter].radius;
        }
        else
        {
            result[contour_id].cx = iter_result[max_iter].cx;
            result[contour_id].cy = iter_result[max_iter].cy;
            result[contour_id].radius = iter_result[max_iter].radius;
        }
    }
}

__host__
void launch_ransac_kernels(int** points_x, int** points_y, ransac_common_params* common_params,
                            ransac_contour_params* contour_params, ransac_result* result)
{
    ransac_common_params* dev_common_params;
    ransac_contour_params* dev_contour_params;
    ransac_result* dev_ransac_result;
    int** dev_points_x;
    int** dev_points_y;
    int** dummy_points_x = (int* *)malloc(sizeof(int*)*common_params->num_contours);
    int** dummy_points_y = (int* *)malloc(sizeof(int*)*common_params->num_contours);
    int i;
    printf("CPU Max Iter : %d\n",  common_params->num_iterations);

    /**
     * Memory Allocations on GPU
     */
    printf("CPU maxptsep : %f\n", common_params->max_point_separation);
    cudaMalloc((void **)&dev_common_params, sizeof(ransac_common_params));
    cudaMalloc((void **)&dev_contour_params, common_params->num_contours*sizeof(ransac_contour_params));
    cudaMalloc((void **)&dev_ransac_result, common_params->num_contours*sizeof(ransac_result));
    
    cudaMalloc((void **)&dev_points_x, common_params->num_contours*sizeof(int *));
    cudaMalloc((void **)&dev_points_y, common_params->num_contours*sizeof(int *));
    
    for (i = 0; i < common_params->num_contours; i++)
    {
        cudaMalloc(&dummy_points_x[i], contour_params[i].consensus_size*sizeof(int));
        cudaMalloc(&dummy_points_y[i], contour_params[i].consensus_size*sizeof(int));
    }
    cudaMemcpy(dev_points_x, dummy_points_x, common_params->num_contours*sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_points_y, dummy_points_y, common_params->num_contours*sizeof(int *), cudaMemcpyHostToDevice);

    /**
     * Memory Copies to device
     */
    cudaMemcpy(dev_common_params, common_params, sizeof(ransac_common_params), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_contour_params, contour_params, common_params->num_contours*sizeof(ransac_contour_params), cudaMemcpyHostToDevice);
    for (i = 0; i < common_params->num_contours; i++)
    {
        cudaMemcpy(dummy_points_x[i], points_x[i], contour_params[i].consensus_size*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dummy_points_y[i], points_y[i], contour_params[i].consensus_size*sizeof(int), cudaMemcpyHostToDevice);
    }
    
    printf("%d %d %f\n", result->cx, result->cy, result->radius);
    if (cudaSuccess == cudaMemcpy(result, dev_ransac_result, common_params->num_contours*sizeof(ransac_result), cudaMemcpyDeviceToHost))
    {
        //printf("AAAAAAAAAAAA\n");
    }
    //printf("%d %d %f\n", result->cx, result->cy, result->radius);
    printf("Number of contours and iter %d %d \n", common_params->num_contours, common_params->num_iterations);
    ransac_kernel<<<common_params->num_contours, common_params->num_iterations>>>(dev_points_x, dev_points_y,
            dev_common_params, dev_contour_params, dev_ransac_result);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
    
//    printf("%d %d %f\n", result->cx, result->cy, result->radius);
    if (cudaSuccess == cudaMemcpy(result, dev_ransac_result, common_params->num_contours*sizeof(ransac_result), cudaMemcpyDeviceToHost))
    {
      //  printf ("BBBBBB\n");
    }
     err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
    printf("Center and Radius %d %d %f", result[0].cx, result[0].cy, result[0].radius);
    
    /**
     * Memory Free on GPU
     */
    for (i = 0; i < common_params->num_contours; i++)
    {
        cudaFree(dummy_points_x[i]);
        cudaFree(dummy_points_y[i]);
    }
     err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
    free(dummy_points_x);
    free(dummy_points_y);
    cudaFree(dev_points_x);
    cudaFree(dev_points_y);
    cudaFree(dev_common_params);
    cudaFree(dev_contour_params);
    cudaFree(dev_ransac_result);
     err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

    return;
}
