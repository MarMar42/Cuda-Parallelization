#include <stdio.h>
#include <cuda_runtime.h>
#define block_size 8
#define N (1<<9)
#define tile_size 64

float* fillArray(float* arr)
{
    //Seed rand()
    srand(42);
    for (int i = 0; i < N*N; i++)
    {
        arr[i] = rand() % 100;
    }
    return arr;
}

void printArray(float* arr)
{
    for (int i = 0; i < N*N; i++)
        printf("%f ", arr[i]);
    printf("\n");
}

float* seq_Transpose(float* arr, float* out)
{
    int row, col, t;
    for (int i = 0; i < N*N; i++)
    {
        row = i/N;
        col = i % N;
        t = col*N + row;
        out[t] = arr[i];
    }
    return out;
}

bool checkTranspose(float* arr, float* arrT)
{
    int row, col, indx, indxT;
    for (int i = 0; i < N*N; i++)
    {
        row = i/N;
        col = i % N;
        indx = row*N + col;
        indxT = col*N + row;
        if (arr[indx] != arrT[indxT])
        {
            return 0;
        }
    }
    return 1;
}
void checkCUDAError(const char *msg) 
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
__global__ void glob_gpuTranspose(float* arr, float* out)
{
    int row, col, t;
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < (N*N))
    {
        row = i/N;
        col = i % N;
        t = col*N + row;
        out[t] = arr[i];
    }
}


__global__ void share_gpuTranspose(float* arr, float* out)
{
    __shared__ float tile[tile_size][tile_size];
 
    unsigned int row = threadIdx.x + blockIdx.x*tile_size;
    unsigned int col = threadIdx.y + blockIdx.y*tile_size;

    for (int i = 0; i < tile_size; i += block_size)
    {
        if ((row < N*N) && (col < N*N))
        tile[threadIdx.y+i][threadIdx.x] = arr[(col+i)*N + row];
    }
    __syncthreads();
    
    row = blockIdx.y*tile_size + threadIdx.x;
    col = blockIdx.x*tile_size + threadIdx.y;
 

    for (int i = 0; i < tile_size; i += block_size)
    {
        if ((row < N*N) && (col < N*N))
        out[(col +i)*N + row] = tile[threadIdx.x][threadIdx.y + i];
    }

 
}

int main (void)
{
    printf("-------------------N = %d----------------\n", N);
    //Declare host arrays
    float* h_in;
    float* h_out;
    //Declare device arrays
    float* d_in;
    float* d_out;
 
    //Set amount of memory required for arrays
    const int num_bytes = N*N*sizeof(float);
 
    //Allocate memory for host array
    h_in = (float*)malloc(num_bytes);
    h_out = (float*)malloc(num_bytes);

    //Fill host array
    h_in = fillArray(h_in);

    //Allocate memory to device
    cudaMalloc((void**) &d_in, num_bytes);
    cudaMalloc((void**) &d_out, num_bytes);

    if(h_in == 0 || h_out == 0 || d_in == 0 || d_out == 0)
    {
        printf("Couldn't allocate memory\n");
        return 1;
    }
 
    //Initialise Timing(Sequential)
    cudaEvent_t launch_begin, launch_end;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);

    cudaEventRecord(launch_begin,0);
    //Sequential Transpose
    h_out = seq_Transpose(h_in, h_out);

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    float seq_time = 0;
	cudaEventElapsedTime(&seq_time, launch_begin, launch_end);
    //End Timing (Sequential)
    float cputp = 1e-9*N*N/(seq_time*1e-3);//divive by 10^9 for giga and times by 10^3 to get seconds
    if (checkTranspose(h_in,h_out))
    {
        printf("CPU: Run Time: %f ms\n", seq_time);
        printf("CPU: Speed Up: %fx\n", seq_time/seq_time);
        printf("CPU: Throughputs: %f GFLOP/s\n\n",cputp );
    }
    else
    printf("Transpose failed (Sequential)");
/////////////////////GPU Section/////////////////////////////
 
    //Copy host arrays to device arrays
    cudaMemcpy(d_in, h_in, num_bytes, cudaMemcpyHostToDevice);
    
    //Declare size of blocks and grid
    size_t num_threads = 256;
    size_t num_blocks = (N*N) / num_threads;
 
    //Zero output memory
    memset(h_out,  0, num_bytes);
    
    if (N % num_threads)
        ++num_blocks;

    //Initialise Timing(Global)
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
 
    cudaEventRecord(launch_begin,0);

    //Global kernel
    glob_gpuTranspose<<<num_blocks, num_threads>>>(d_in, d_out);

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    float glob_time = 0;
	cudaEventElapsedTime(&glob_time, launch_begin, launch_end);
    //End Timing
 
    cudaMemcpy(h_out, d_out, num_bytes, cudaMemcpyDeviceToHost);
    
    if (checkTranspose(h_in,h_out))
    {
        printf("GPU(Global): Run Time: %f ms\n", glob_time);
        printf("GPU(Global): Speed Up: %fx\n", seq_time/glob_time);
        printf("GPU(Global): Throughputs: %f GFLOP/s\n", 1e-9*N*N/(glob_time*1e-3));
        printf("GPU(Global): Ratio of Throughputs: %f\n\n", 1e-9*N*N/(glob_time*1e-3)/cputp);
    }
    else
    printf("Transpose failed (Global)");

    checkCUDAError("glob_gpuTranspose");

    dim3 dimGrid(N/tile_size,N/tile_size,1);
    dim3 dimBlock(tile_size, block_size,1);
 
    memset(h_out,  0, num_bytes);
    cudaMemset(d_out, 0, num_bytes);

    //Initialise Timing(Shared)
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
 
    cudaEventRecord(launch_begin,0);

    //Shared kernel    
    share_gpuTranspose<<<dimGrid, dimBlock>>>(d_in, d_out);

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    float share_time = 0;
	cudaEventElapsedTime(&share_time, launch_begin, launch_end);
    //End Timing
 
    cudaMemcpy(h_out, d_out, num_bytes, cudaMemcpyDeviceToHost);
    
    if (checkTranspose(h_in,h_out))
    {
        printf("GPU (Shared) tile size = %d, block size = %d: Run Time: %f ms\n", tile_size, block_size, share_time);
        printf("GPU (Shared) tile size = %d, block size = %d: Speed Up: %fx\n", tile_size, block_size, seq_time/share_time);
        printf("GPU (Shared) tile size = %d, block size = %d: Throughputs: %f GFLOPS/s\n\n", tile_size, block_size, 1e-9*N*N/(share_time*1e-3));
        printf("GPU (Shared) tile size = %d, block size = %d: Throughput Ratio: %f GFLOPS/s\n\n", tile_size, block_size, 1e-9*N*N/(share_time*1e-3)/cputp);
    }
    else
    printf("Transpose failed (Shared)");

    checkCUDAError("share_gpuTranspose");

    cudaEventDestroy(launch_begin);
    cudaEventDestroy(launch_end);
    //printArray(h_out);
    //Free host array
    free(h_in);
    free(h_out);
 
    //Free device array
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}