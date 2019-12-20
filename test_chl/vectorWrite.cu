#include <cuda.h>
#include "cuda_runtime.h"
// #include <cutil.h>
#include "texture_fetch_functions.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>

#include <stdio.h>
#include <iostream>


#define DATATYPE int
#define ARRAYLEN 128*1024*1024

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		std::cerr << std::endl << " CUDA error   " << file
                  << "(" << line << ")" << " : " << errorMessage
                  << " -> " << cudaGetErrorString(err) << "(" << (int) err
                  << ") "<< std::endl << std::endl;
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}
}

#define __CUDA_ERROR(msg)                                                       \
                    {                                                           \
                        cudaDeviceSynchronize();                                \
                        __getLastCudaError (msg, __FILE__, __LINE__);\
                    }



#define repeat128(x) for(int i=0;i<128;i++){x}

typedef int mytype;
mytype A_val = 1;
mytype B_val = 2;

__global__ void matrix_add_4(const mytype * __restrict__ A, const mytype * __restrict__ B, mytype * __restrict__ C, const size_t len){
    int start = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = gridDim.x* blockDim.x *4;
    int4* A4 = (int4*)A;
    int4* B4 = (int4*)B;
    int4* C4 = (int4*)C;
    for(int i=start*4;i<len;i+=stride)
    {
        C4[i].x = A4[i].x + B4[i].x;
        C4[i].y = A4[i].y + B4[i].y;
        C4[i].z = A4[i].z + B4[i].z;
        C4[i].w = A4[i].w + B4[i].w;
    }
}

template<int size>
__global__ void matrix_add(const mytype * __restrict__ A, const mytype * __restrict__ B, mytype * __restrict__ C, const size_t len){
    int start = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = gridDim.x* blockDim.x * size;
    for(int i=start*size;i<len;i+=stride)
    {
#pragma unroll
        for(int j=0;j<size;j++)
        {   
            mytype x = A[i+j]+B[i+j];
            C[i+j] = x;
        }
    }
}

int main(){


//   int device;
//   cudaGetDevice(&device);
  cudaSetDevice(5);
    mytype *A,*B,*C;
  size_t len = ARRAYLEN;
  cudaError_t err = cudaMallocManaged(&A, ARRAYLEN * sizeof(mytype));
//   if (err != cudaSuccess) {std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; return 0;}
  cudaMallocManaged(&B, ARRAYLEN*sizeof(mytype));
  cudaMallocManaged(&C, ARRAYLEN*sizeof(mytype));
  for (int x = 0; x < len; x++){
      A[x] = A_val;
      B[x] = B_val;
      C[x] = 0;
    }
  int attr = 0;
  cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentManagedAccess,0);
  if (attr){
    cudaMemPrefetchAsync(A, len, 0);
    cudaMemPrefetchAsync(B, len, 0);
    cudaMemPrefetchAsync(C, len, 0);}
//   dim3 threads(32,32);
//   dim3 blocks((size_w+threads.x-1)/threads.x, (size_h+threads.y-1)/threads.y);
//   matrix_add_2D<<<blocks,threads>>>(A,B,C, size_w, size_h);


    cudaEvent_t s1,s2;
    cudaEventCreate(&s1);
    cudaEventCreate(&s2);
    cudaEventRecord(s1,0);
    cudaEventSynchronize(s1);
    matrix_add<1><<<256*4,128>>>(A,B,C,len);
    matrix_add<2><<<256*4,128>>>(A,B,C,len);
    matrix_add<4><<<256*4,128>>>(A,B,C,len);
    matrix_add<8><<<256*4,128>>>(A,B,C,len);
    matrix_add<16><<<256*4,128>>>(A,B,C,len);
    matrix_add<32><<<256*4,128>>>(A,B,C,len);
    // matrix_add_4<<<256*4,128>>>(A,B,C,len);
    cudaEventRecord(s2,0);
    cudaEventSynchronize(s2);
    float time1;
    cudaEventElapsedTime(&time1,s1,s2);
    cudaDeviceSynchronize();
    cudaEventDestroy(s1);
    cudaEventDestroy(s2);
    printf("%f\n",time1);

  err = cudaGetLastError();
  if (err != cudaSuccess) {std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; return 0;}
//   for (int x = 0; x < size_h; x++)
//     for (int y = 0; y < size_w; y++)
//       if (C[x][y] != A_val+B_val) {std::cout << "mismatch at: " << x << "," << y << " was: "  << C[x][y] << " should be: " << A_val+B_val << std::endl; return 0;} ;
  std::cout << "Success!" << std::endl;
  return 0;
}