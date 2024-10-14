#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

#define  MAXIMUM_VALUE   1000000
#define  BLOCK  256 // Threads per block

// Device functions for performing atomic operations on float values.
__device__ float atomicMinFLT (float* addr, float val)
{
    float former;
    former =  __int_as_float(atomicMin((int *)addr, __float_as_int(val)));

    return former;
};

__device__ float atomicMaxFLT (float* addr, float val)
{
    float former;
    former = __int_as_float(atomicMax((int *)addr, __float_as_int(val)));

    return former;
};


/*
   Code adapted from:
   https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/
   03_sum_reduction/diverged/sumReduction.cu
*/
__global__ void ArrayStats(float* array, float* sum, float* min, float* max, float* sqr_sum, int n)
{
  /*
    This kernel finds the minimum and maximum values of the given array.
    At this same time, it calculates the sum and sum of squares,
    both needed to calculate the standard deviation.
  */

  __shared__ float min_val[BLOCK];
  __shared__ float max_val[BLOCK];
  __shared__ float sum_val[BLOCK];
  __shared__ float sqr_val[BLOCK]; // Sum of squares

  int stride = blockDim.x * gridDim.x;
  int idx =  (blockIdx.x * blockDim.x) + threadIdx.x;

  // Initialize local variables
  float local_min = FLT_MAX;
  float local_max = 0.0f;
  float local_sum = 0.0f;
  float local_sqr = 0.0f;

  // Calculate local values
  for(int s = idx; s < n; s += stride){

    local_sum += array[s];
    local_sqr += array[s] * array[s];
    local_min = fminf(local_min, array[s]);
    local_max = fmaxf(local_max, array[s]);
  }

  // Add local values to shared memory.
  min_val[threadIdx.x] = local_min;
  max_val[threadIdx.x] = local_max;
  sum_val[threadIdx.x] = local_sum;
  sqr_val[threadIdx.x] = local_sqr;

  __syncthreads();

  // Perform reductions
  for(int s = blockDim.x / 2; s > 0; s >>= 1){

    if(threadIdx.x < s)
    {
      sum_val[threadIdx.x] += sum_val[threadIdx.x + s];
      sqr_val[threadIdx.x] += sqr_val[threadIdx.x + s];
      min_val[threadIdx.x] = fminf(min_val[threadIdx.x], min_val[threadIdx.x + s]);
      max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
    }
    __syncthreads();
  }

  // Pass final values to global memory.
  if(threadIdx.x == 0)
  {
     atomicMinFLT(min,  min_val[0]);
     atomicMaxFLT(max,  max_val[0]);
     atomicAdd(sum,     sum_val[0]);
     atomicAdd(sqr_sum, sqr_val[0]);
  }

}


int main(int argc, char* argv[]){

  float *dmin, *dmax, *dsum, *dsqr_sum, *darray; // Device pointers

  if( argc < 3 ) {
    printf( "Format: stats_s <size of array> <random seed>\n" );
    printf( "Arguments:\n" );
    printf( "  size of array - This is the size of the array to be generated and processed\n" );
    printf( "  random seed   - This integer will be used to seed the random number\n" );
    printf( "                  generator that will generate the contents of the array\n" );
    printf( "                  to be processed\n" );

    exit( 1 );
  }


  // Get the array size and random seed.
  unsigned int n, seed;
  n = atoi( argv[1] );
  seed = atoi( argv[2] );


  // Allocate array for host.
  float *array = (float *)malloc(n * sizeof(float));


  // Initialize array with random values.
  srand( seed );
  for(unsigned int i = 0; i < n; i++ )
  {
    array[i] = ( (double) rand() / (double) RAND_MAX ) * MAXIMUM_VALUE;
  }


  // Allocate memory on device
  cudaMalloc(&darray,   sizeof(float) * n);
  cudaMalloc(&dmin,     sizeof(float));
  cudaMalloc(&dmax,     sizeof(float));
  cudaMalloc(&dsum,     sizeof(float));
  cudaMalloc(&dsqr_sum, sizeof(float));


  // Copy host variables to device
  float min = FLT_MAX, max = 0.0f, sum = 0.0f,  sqr_sum = 0.0f;
  cudaMemcpy(darray,   array, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dmin,     &min, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dmax,     &max, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dsum,     &sum, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dsqr_sum, &sqr_sum, sizeof(float), cudaMemcpyHostToDevice);


  // Start timer.
  struct timeval start, end;
  gettimeofday( &start, NULL );


  // Declare grid size
  int gridsize = (n + BLOCK - 1) / BLOCK;


  // Launch the kernel
  ArrayStats<<<gridsize, BLOCK>>>(darray, dsum, dmin, dmax, dsqr_sum, n);

  cudaDeviceSynchronize();

  // Copy the results back to the host
  cudaMemcpy(&sum,     dsum, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&min,     dmin, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max,     dmax, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sqr_sum, dsqr_sum, sizeof(float), cudaMemcpyDeviceToHost);

  // Calculate mean.
  float mean = (sum / n);

  // Calculate standard deviation.
  float stddev = sqrtf((sqr_sum / n) - powf(mean, 2.0f));

  // End timer
  gettimeofday( &end, NULL );

  // Calculate the runtime.
  double runtime;
  runtime = ( ( end.tv_sec  - start.tv_sec ) * 1000.0 ) + ( ( end.tv_usec - start.tv_usec ) / 1000.0 );


  // Display results.
  printf( "Statistics for array ( %d, %d ):\n", n, seed );
  printf( "    Minimum = %4.6f, Maximum = %4.6f\n", min, max );
  printf( "    Mean = %4.6f, Standard Deviation = %4.6f\n", mean, stddev );
  printf( "Processing Time: %4.4f milliseconds\n", runtime );

  // Free the allocated memory.
  cudaFree(darray);
  cudaFree(dmin);
  cudaFree(dmax);
  cudaFree(dsum);
  cudaFree(dsqr_sum);

  free(array);

  return 0;

}
