#include <stdio.h>
#include <cuda_runtime.h>

#define num_threads 256
#define N (1<<20)
void fillArray(float* arr)
{
    //Seed rand()
    srand(42);
    for (int i = 0; i < N/2; i++)
    {
        arr[i] = rand() % 10;
    }
 ///Negatives
    for (int i = N/2; i < N; i++)
    {
        arr[i] = -1*(rand() % 10);
    }
}

void printArray(float* arr)
{
    for (int i = 0; i < N; i++)
        printf("%f ", arr[i]);
    printf("\n");
}

void seq_VecAdd(float* arr_in, float* ans)
{
    float total = 0;
    for (int i = 0; i < N; i++)
    {
        total += arr_in[i];
    }
    *ans = total;
}


__global__ void share_VecAdd(float* arr_in, float* arr_out)
{
    __shared__ float sVec[num_threads];
 
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    
    //Loading shared memory
    if (i < N)
    {
    sVec[threadIdx.x] = arr_in[i];
    }
    __syncthreads();
    
    //reduction in shared mem
    for(unsigned int j = blockDim.x/2; j > 0; j >>= 1)
    {
        if (threadIdx.x < j)
        {
            sVec[threadIdx.x] += sVec[threadIdx.x + j];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(arr_out,sVec[threadIdx.x]);
    }
    
}

__global__ void glob_VecAdd(float* arr_in, float* arr_out)
{
    
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
 
    for(unsigned int j = blockDim.x/2; j > 0; j = j/2)
    {
        if (threadIdx.x < j)
        {
            arr_in[i] += arr_in[i + j];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(arr_out,arr_in[i]);
    }
    
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
    float ans;
    //Declare host array
    float* arr_h;
    float* out_h;
 
    //Declare device array
    float* arr_in_d;
    float* arr_out_d;
 
    //Set amount of memory required for arrays
    const int num_bytes = N*sizeof(float);
 
    //Allocate memory for host array
    arr_h = (float*)malloc(num_bytes);
    out_h = (float*)malloc(num_bytes);

    //Allocate memory to the device
    cudaMalloc((void**)&arr_in_d, num_bytes);
    cudaMalloc((void**)&arr_out_d, num_bytes);

    //Fill host arrays with random values
    fillArray(arr_h);


//////Sequential////////
     ////Start Timing
    cudaEvent_t launch_begin, launch_end;
 
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
 
    cudaEventRecord(launch_begin,0);
 
    //Sequential calculation for checking
    seq_VecAdd(arr_h, &ans);
 
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    float seq_time = 0;
	cudaEventElapsedTime(&seq_time, launch_begin, launch_end);
    float cputp = 1e-9*N/(seq_time*1e-3); //divive by 10^9 for giga and times by 10^3 to get seconds
    printf("CPU: Run Time: %f ms\n", seq_time);
    printf("CPU: Speed Up: %fx\n", seq_time/seq_time);
    printf("CPU: Throughputs: %f GFLOP/s\n\n",cputp );
 //////////////GPU Shared///////////////////
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
 
    int num_blocks = N/num_threads;
 
    if (N % num_threads)
        num_blocks++;
    
    cudaEventRecord(launch_begin,0);
 
    cudaMemcpy(arr_in_d, arr_h, num_bytes, cudaMemcpyHostToDevice);
    
    share_VecAdd<<<num_blocks,num_threads>>>(arr_in_d, arr_out_d );

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    
    float share_time = 0;
	cudaEventElapsedTime(&share_time, launch_begin, launch_end);
 
    cudaMemcpy(out_h, arr_out_d, num_bytes, cudaMemcpyDeviceToHost);
    checkCUDAError("share_VecAdd");

    if (out_h[0] == ans)
    {
        printf("Number of threads = %d\n\n", num_threads); 
        printf("GPU(Shared) Tile Size = %d: Run Time: %f ms\n", num_threads, share_time);
        printf("GPU(Shared) Tile Size = %d: Speed Up: %fx\n", num_threads, seq_time/share_time);
        printf("GPU(Shared) Tile Size = %d: Throughputs: %f GFLOP/s\n", num_threads, 1e-9*N/(share_time*1e-3));
        printf("GPU(Shared) Tile Size = %d: Ratio of Throughputs: %f\n\n", num_threads, 1e-9*N/(share_time*1e-3)/cputp);
    }
    else
    {
        printf("GPU Shared Failed : %f\n", out_h[0]);
    }
/////////////GPU Global/////////////////
    cudaMemset(arr_out_d, 0, num_bytes);
    
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
 
    if (N % num_threads)
        num_blocks++;
    
    cudaEventRecord(launch_begin,0);
 
    cudaMemcpy(arr_in_d, arr_h, num_bytes, cudaMemcpyHostToDevice);
    
    glob_VecAdd<<<num_blocks,num_threads>>>(arr_in_d, arr_out_d);

    
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    
    float glob_time = 0;
	cudaEventElapsedTime(&glob_time, launch_begin, launch_end);
 
    cudaMemcpy(out_h, arr_out_d, num_bytes, cudaMemcpyDeviceToHost);
    checkCUDAError("glob_VecAdd");

    if (out_h[0] == ans)
    {
        printf("GPU(Global): Run Time: %f ms\n", glob_time);
        printf("GPU(Global): Speed Up: %fx\n", seq_time/glob_time);
        printf("GPU(Global): Throughputs: %f GFLOP/s\n", 1e-9*N/(glob_time*1e-3));
        printf("GPU(Global): Ratio of Throughputs: %f\n\n", 1e-9*N/(glob_time*1e-3)/cputp);
    }
    else
    {
        printf("GPU Global Failed : %f\n", out_h[0]);
    }


    //Free host array
    free(arr_h);
    free(out_h);
 
    //Free device arrays
    cudaFree(arr_in_d);
    cudaFree(arr_out_d);


    return 0;
}