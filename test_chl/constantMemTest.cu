#include <cuda.h>
#include "cuda_runtime.h"
// #include <cutil.h>
#include "texture_fetch_functions.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>

#include <stdio.h>
#include <iostream>
using namespace std;

#define DATATYPE int
#define ARRAYLEN 128 * 1024 * 1024

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        std::cerr << std::endl
                  << " CUDA error   " << file
                  << "(" << line << ")"
                  << " : " << errorMessage
                  << " -> " << cudaGetErrorString(err) << "(" << (int)err
                  << ") " << std::endl
                  << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
}

#define __CUDA_ERROR(msg)                            \
    {                                                \
        cudaDeviceSynchronize();                     \
        __getLastCudaError(msg, __FILE__, __LINE__); \
    }

#define size 100

__constant__ int *prt[size];

__global__ void matrix_add()
{
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        *(prt[i]) = 2;
    }
}

int main()
{

    //   int device;
    //   cudaGetDevice(&device);
    cudaSetDevice(5);
    int *p = (int *)malloc(sizeof(int) * size);
    printf("host:%d,%d\n", sizeof(int *), sizeof(int));
    // int *pd[size];
    // for (int i = 0; i < size; i++)
    // {
    //     cudaMalloc(&(pd[i]), sizeof(int) * 1);
    // }
    int *pd;
    cudaMalloc(&pd, sizeof(int) * size);

    int *pdarray[size];
    for (int i = 0; i < size; i++)
    {
        pdarray[i] = pd + i;
    }
    cudaMemcpyToSymbol(prt, pdarray, sizeof(int *) * size);
    matrix_add<<<1, 256>>>();
    //  __CUDA_ERROR("dfa");
    cudaMemcpy(p, pd, sizeof(int) * size, cudaMemcpyDeviceToHost);
    // cudaMemcpyFromSymbol(hd, data, sizeof(int) * 1024);
    // for (int i = 0; i < size; i++)
    // {
    //     cudaMemcpy(&p[i], pd[i], sizeof(int) * 1, cudaMemcpyDeviceToHost);
    // }
    for (int i = 0; i < size; i++)
    {
        cout << p[i] << " ";
    }
}