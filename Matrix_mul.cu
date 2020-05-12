#include <stdio.h>
#include <cuda_runtime.h>

#define tile_size 32
#define N (1<<9)
///////////Functions that check if matrix was multiplied
void matrix_multiply_seq(float *a, float *b, float *ab, size_t width){
	int i, j, k;
	for(i=0; i<width; i++)
		for(j=0; j<width; j++){
			ab[i*width+j]=0.0;
			for(k=0; k<width; k++){
				ab[i*width+j] += a[i*width+k] * b[k*width+j];
			}
		}
}

int matrixEqual(float *matrixA, float *matrixB){
  for ( int r = 0; r < N  ; r++ )
    for ( int c = 0; c < N ; c++ ){
      if ( matrixA[r*N+c] != matrixB[r*N+c] ){
        return 1;
      }
    }
  return 0;
}
///////////
void fillArray(float* arr)
{
    //Seed rand()
    srand(42);
    for (int i = 0; i < N*N; i++)
    {
        arr[i] = rand() % 100;
    }
}

void printArray(float* arr)
{
    for (int i = 0; i < N*N; i++)
        printf("%f ", arr[i]);
    printf("\n");
}

__global__ void matrixMul(float* a, float* b, float* c)
{
    __shared__ float a_s[tile_size][tile_size];
    __shared__ float b_s[tile_size][tile_size];
 
    int row = threadIdx.y + blockIdx.y*tile_size;
    int col = threadIdx.x + blockIdx.x*tile_size;
 
    float sum_value = 0;
    
        for (int i = 0; i < N/tile_size; i++)
        {
            a_s[threadIdx.y][threadIdx.x] = a[row*N + (i*tile_size + threadIdx.x)];
            b_s[threadIdx.y][threadIdx.x] = b[col + (i*tile_size + threadIdx.y)*N];
            __syncthreads();

            for (int j = 0; j < tile_size; j++)
            {
                sum_value += a_s[threadIdx.y][j]*b_s[j][threadIdx.x];
            }
         __syncthreads();
        }
    c[row*N+col] = sum_value;
}

void checkCUDAError(const char *msg) 
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

int main (void)
{
    printf("-------------------N = %d----------------\n", N);
    //Declare host arrays
    float* h_a;
    float* h_b;
    float* h_c;
    float* h_ans;
 
    //Declare device arrays
    float* d_a;
    float* d_b;
    float* d_c;
 
    //Set amount of memory required for arrays
    const int num_bytes = N*N*sizeof(float);
 
    //Allocate memory for host array
    h_a = (float*)malloc(num_bytes);
    h_b = (float*)malloc(num_bytes);
    h_c = (float*)malloc(num_bytes);
    h_ans = (float*)malloc(num_bytes);
 
    //Allocate memory to the device
    cudaMalloc((void**)&d_a, num_bytes);
    cudaMalloc((void**)&d_b, num_bytes);
    cudaMalloc((void**)&d_c, num_bytes);

    //Fill host arrays with random values
    fillArray(h_a);
    fillArray(h_b);


    ////Start Timing
    cudaEvent_t launch_begin, launch_end;
 
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
 
    cudaEventRecord(launch_begin,0);
 
    //Sequential calculation for checking
    matrix_multiply_seq(h_a,h_b,h_ans,N);
 
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    float seq_time = 0;
	cudaEventElapsedTime(&seq_time, launch_begin, launch_end);
    cudaEventSynchronize(launch_end);
 
    float cputp = 2*1e-9*N/(seq_time*1e-3); //divive by 10^9 for giga and times by 10^3 to get seconds

    printf("CPU: Run Time: %f ms\n", seq_time);
    printf("CPU: Speed Up: %fx\n", seq_time/seq_time);
    printf("CPU: Throughputs: %f GFLOP/s\n\n",cputp );
//////////GPU Section///////////////////////
    //Copy host arrays to device arrays
    cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_bytes, cudaMemcpyHostToDevice);
 
    ////////Start timing
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    
    dim3 dimBlock(tile_size, tile_size,1);
    dim3 dimGrid(N/tile_size,N/tile_size,1);

    cudaEventRecord(launch_begin,0);
    matrixMul<<<dimGrid,dimBlock>>>(d_a,d_b,d_c);

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    float share_time = 0;
	cudaEventElapsedTime(&share_time, launch_begin, launch_end);
    ////////End Timing
    cudaMemcpy(h_c, d_c, num_bytes, cudaMemcpyDeviceToHost);


    if(matrixEqual(h_c, h_ans) == 0)
    {
        printf("GPU(Shared) Tile Size = %d: Run Time: %f ms\n",tile_size, share_time);
        printf("GPU(Shared) Tile Size = %d: Speed Up: %fx\n", tile_size, seq_time/share_time);
        printf("GPU(Shared) Tile Size = %d: Throughputs: %f GFLOP/s\n", tile_size, 2*1e-9*N/(share_time*1e-3));
        printf("GPU(Shared) Tile Size = %d: Ratio of Throughputs: %f\n\n", tile_size, 2*1e-9*N/(share_time*1e-3)/cputp);
    }
    else {
	    printf("Verification failed.\n");
        checkCUDAError("share_gpuMatMul");
    }


    //printArray(h_ans);
    //printArray(h_c);
    //Free host arrays
    free(h_a);
    free(h_b);
    free(h_c);
 
    //Free device arrays
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}