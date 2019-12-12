#include <cuda.h>
#include "cuda_runtime.h"
// #include <cutil.h>
#include "texture_fetch_functions.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
#include <time.h>

#include <stdio.h>
#include <iostream>

using namespace cooperative_groups;


typedef int mytype;
#define ARRAYLEN 32 * 1000

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


template<int VW_SIZE,typename T>
__device__ __forceinline__
void VWWrite(thread_block_tile<VW_SIZE>& tile,int *pAllsize,T* writeStartAddr,
        const int& writeCount,T* data)
{
    int sum = 0;
    int bias = 0;
    VWInclusiveScanAdd<VW_SIZE,int>(tile,writeCount,sum);
    if(tile.thread_rank() == tile.size()-1 && sum !=0)
    {
        bias = atomicAdd(pAllsize,sum);
    }
    bias = tile.shfl(bias,tile.size()-1);
    sum -= writeCount;
    for(int it = 0;it<writeCount;it++)
    {
        *(writeStartAddr+bias+sum+it) = data[it];   
    }
}

__host__ __device__ __forceinline__ int is_write(mytype v)
{
    return !(v%10);
}

#define REG 6

__global__ void write_test1(const mytype *f1,int f1Size, mytype* f2, int* pf2Size,long long int  *st,long long int  *ed)
{
    __shared__ mytype array[REG * 512];
    int k=0;
    long long int  start = 0,end = 0;
    thread_block g = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(g);
    if(tile.thread_rank()==0)
        start = clock64();
    int id = threadIdx.x;
    int stride = blockDim.x;
    int warp_id = threadIdx.x / 32;
    int bwid =  16 * blockIdx.x + warp_id;

    int count = warp_id<1?8:warp_id;
    tile.sync();
    for(int i=0;i<count;i++)
    {
        for(id = threadIdx.x;id<f1Size;id=id+stride)
        {
            mytype v = f1[id];
            if(is_write(v))
            {
                array[threadIdx.x * REG + k++] = v;
            }
            if(tile.any(k>=REG))
            {           
                VWWrite<32,mytype>(tile, pf2Size, f2, k , array + threadIdx.x * REG);
                k=0;
            }       
        }
    }
    VWWrite<32,mytype>(tile, pf2Size, f2, k , array + threadIdx.x * REG);
    if(tile.thread_rank()==0)
        end = clock64();
    if(tile.thread_rank() == 0)
    {
        st[bwid] = start;
        ed[bwid] = end;
    }

}

int main()
{
    mytype *f1,*f2;
    int *f2Size;
    long long int  *st,*ed;
    srand( (unsigned)time( NULL ) );  
    cudaSetDevice(2);
    size_t ds = ARRAYLEN*sizeof(mytype);
    cudaMallocManaged(&f1, ds);
    cudaMallocManaged(&f2, ds * 32*32);
    cudaMallocManaged(&f2Size, sizeof(int));
    cudaMallocManaged(&st, 512 * 32 * sizeof(long long int ));
    cudaMallocManaged(&ed, 512 * 32 * sizeof(long long int  ));
    for(int i=0;i<ARRAYLEN;i++)
    {
        f1[i] = rand()%1000;
    }
    int attr = 0;
    cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentManagedAccess,0);
    if (attr)
    {
        cudaMemPrefetchAsync(f1, ds, 0);
        // cudaMemPrefetchAsync(f2, ds, 0);
    }
    cudaEvent_t s1,s2;
    cudaEventCreate(&s1);
    cudaEventCreate(&s2);
    cudaEventRecord(s1,0);
    cudaEventSynchronize(s1);
    write_test1<<<68*2,512>>>(f1,ARRAYLEN,f2,f2Size,st,ed);
    cudaEventRecord(s2,0);
    cudaEventSynchronize(s2);
    float time1;
    cudaEventElapsedTime(&time1,s1,s2);
    cudaDeviceSynchronize();
    cudaEventDestroy(s1);
    cudaEventDestroy(s2);
    printf("%f\n",time1);   
    cudaDeviceSynchronize();

    for(int i=0;i<68*2*16;i++)
    {
        std::cout<<i<<"\t"<<st[i]<<"\t"<<ed[i]<<"\t"<<ed[i]-st[i]<<"\n";
    }

    __CUDA_ERROR("");
    return 0;
}