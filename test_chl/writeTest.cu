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
#define ARRAYLEN 32 * 10000000

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
void VWInclusiveScanAdd(thread_block_tile<VW_SIZE>& tile,const T& value,T& sum)
{
    sum = value;
    for(int i=1;i<=tile.size()/2;i*=2)
    {
        T n = tile.shfl_up(sum, i);
        if (tile.thread_rank() >= i)
        {
            sum += n;
        }
    }
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

template<int VW_SIZE,typename T>
__device__ __forceinline__
void VWWrite_v2(thread_block_tile<VW_SIZE>& tile,int *pAllsize,T* writeStartAddr,
        const int& writeCount,T* data)
{
    int sum = 0;
    int bias = 0;
    int all = 0;
    int k=0;
    int count=writeCount;
    VWInclusiveScanAdd<VW_SIZE,int>(tile,writeCount,sum);
    if(tile.thread_rank() == tile.size()-1 && sum !=0)
    {
        bias = atomicAdd(pAllsize,sum);
    }
    bias = tile.shfl(bias,tile.size()-1);
    while(1)
    {
        unsigned mask = tile.ballot(count>0);
        int flag = __ffs(mask);
        if (flag  == 0)
            break;
        if(count > 0)
        {
            unsigned mymask = 1 << (tile.thread_rank()+1) - 1;
            mymask = mask & mymask ;
            int mybias = __ffs(mymask);
            *(writeStartAddr+bias+all+mybias) = data[k++];
            count--;
        }
    }
}

template<int VW_SIZE,typename T>
__device__ __forceinline__
void VWWrite_v3(thread_block_tile<VW_SIZE>& tile,int *pAllsize,T* writeStartAddr,
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
    for(int it = 0;it<8;it++)
    {
        *(writeStartAddr+bias+sum+it) = data[it];     
    }
}

template<int VW_SIZE,typename T>
__device__ __forceinline__
void VWWrite_v4(thread_block_tile<VW_SIZE>& tile,int *pAllsize,T* writeStartAddr,
        const int& writeCount,T* data)
{
    int sum = 0;
    int bias = 0;
    int all = 0;
    int k=0;
    int count=writeCount;
    VWInclusiveScanAdd<VW_SIZE,int>(tile,writeCount,sum);
    if(tile.thread_rank() == tile.size()-1 && sum !=0)
    {
        bias = atomicAdd(pAllsize,sum);
    }
    bias = tile.shfl(bias,tile.size()-1);
    while(1)
    {
        unsigned mask = tile.ballot(count>0);
        int flag = __ffs(mask);
        if (flag  == 0)
            break;
        if(count > 0)
        {
            unsigned mymask = 1 << (tile.thread_rank()+1) - 1;
            mymask = mask & mymask ;
            int mybias = __ffs(mymask);
#pragma unroll
            for(int i=0;i<8;i++)
            {
                if(k == i)
                {
                    *(writeStartAddr+bias+all+mybias) = data[i];
                }
            }
            k++;
            // *(writeStartAddr+bias+all+mybias) = data[k++];
            count--;
        }
    }
}

__host__ __device__ __forceinline__ int is_write(mytype v)
{
    return !(v%4);
}

#define REG 16

__global__ void test(const mytype *f1,int f1Size, mytype* f2, int* pf2Size)
{
    mytype array[8]={0,1,2,3,4,5,6,7};
    thread_block g = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(g);
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(;id<f1Size;id=id+stride)
    {
        array[0]=1;
    }
    int k = 8;
    VWWrite<32,mytype>(tile, pf2Size, f2, k , array);
}

__global__ void write_test1(const mytype *f1,int f1Size, mytype* f2, int* pf2Size)
{
    __shared__ mytype array[REG * 512];
    int k=0;
    thread_block g = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(g);
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(;id<f1Size;id=id+stride)
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
    VWWrite<32,mytype>(tile, pf2Size, f2, k , array + threadIdx.x * REG);
    k=0;
}

__global__ void write_test2(const mytype *f1,int f1Size, mytype* f2, int* pf2Size)
{
    mytype array[REG];
    int k=0;
    thread_block g = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(g);
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(;id<f1Size;id=id+stride)
    {
        mytype v = f1[id];
        if(is_write(v))
        {
            array[k++] = v;
        }
        if(tile.any(k>=REG))
        {           
            VWWrite<32,mytype>(tile, pf2Size, f2, k , array);
            k=0;
        }       
    }
    VWWrite<32,mytype>(tile, pf2Size, f2, k , array);
    k=0;
}

__global__ void write_test3(const mytype *f1,int f1Size, mytype* f2, int* pf2Size)
{
    int k=0;
    mytype array[8];
    thread_block g = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(g);
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(;id<f1Size;id=id+stride)
    {
        mytype v = f1[id];
        int ok = is_write(v);
        if(ok)
        {
            // array[k] = v;
#pragma unroll
            for(int i=0;i<8;i++)
            {
                if(k == i)
                {
                    array[i] = v;
                }
            }
            k++;
        }
        if(tile.any(k>=8))
        {
            VWWrite_v3<32,mytype>(tile, pf2Size, f2, k , array);
            k=0;
        }   
    }
    VWWrite_v3<32,mytype>(tile, pf2Size, f2, k , array);
}

__global__ void write_test4(const mytype *f1,int f1Size, mytype* f2, int* pf2Size)
{
    mytype array[REG];
    int k=0;
    thread_block g = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(g);
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(;id<f1Size;id=id+stride)
    {
        mytype v = f1[id];
        if(is_write(v))
        {
            array[k++] = v;
        }
        if(tile.any(k>=REG))
        {           
            VWWrite_v2<32,mytype>(tile, pf2Size, f2, k , array);
            k=0;
        }       
    }
    VWWrite_v2<32,mytype>(tile, pf2Size, f2, k , array);
    k=0;
}

#define REG5 1024

__global__ void write_test5(const mytype *f1,int f1Size, mytype* f2, int* pf2Size)
{
    __shared__ int count;
    __shared__ mytype array[REG5];
    __shared__ int all;
    if(threadIdx.x == 0)
        count =0;
    __syncthreads();
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(;id<f1Size;id=id+stride)
    {
        mytype v = f1[id];
        if(is_write(v))
        {
            int bias = atomicAdd(&count,1);
            array[bias] = v;
        }
        __syncthreads();
        if(count >= REG5 - 512)
        {
            if(threadIdx.x == 0)
                all = atomicAdd(pf2Size,count);
            __syncthreads();
            if(threadIdx.x < count)
                *(f2+all+threadIdx.x) = array[threadIdx.x];
            if(threadIdx.x == 0)
                count =0;
        } 
    }
    if(threadIdx.x == 0)
        all = atomicAdd(pf2Size,count);
    __syncthreads();
    if(threadIdx.x < count)
        *(f2+all+threadIdx.x) = array[threadIdx.x];
}

typedef void(*func)(const mytype*,int, mytype*, int*);

void test_time(func f,const mytype* f1,mytype *f2,int* f2Size)
{
    cudaEvent_t s1,s2;
    cudaEventCreate(&s1);
    cudaEventCreate(&s2);
    cudaEventRecord(s1,0);
    cudaEventSynchronize(s1);
    f<<<1,512>>>(f1,ARRAYLEN,f2,f2Size);
    cudaEventRecord(s2,0);
    cudaEventSynchronize(s2);
    float time1;
    cudaEventElapsedTime(&time1,s1,s2);
    cudaDeviceSynchronize();
    cudaEventDestroy(s1);
    cudaEventDestroy(s2);
    printf("%f\n",time1);
}
int main()
{
    mytype *f1,*f2;
    int *f2Size;
    srand( (unsigned)time( NULL ) );  
    cudaSetDevice(2);
    size_t ds = ARRAYLEN*sizeof(mytype);
    cudaMallocManaged(&f1, ds);
    cudaMallocManaged(&f2, ds);
    cudaMallocManaged(&f2Size, sizeof(int));
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

    int cT=0;
    for(int i=0;i<ARRAYLEN;i++)
    {
        if(is_write(i))
            cT++;
    }

    f2Size[0] = 0;

    test_time(f1,ARRAYLEN,f2,f2Size);

    f2Size[0] = 0;
    
    for(int i=0;i<1;i++)
    {
        f2Size[0] = 0;
        test_time(write_test5,f1,f2,f2Size);
        __CUDA_ERROR("");
    }

    f2Size[0] = 0;
    
    for(int i=0;i<1;i++)
    {
        f2Size[0] = 0;
        test_time(write_test4,f1,f2,f2Size);
    }
    printf("%d,%d\n",f2Size[0],cT);
    
    printf("test2\n");
    for(int i=0;i<1;i++)
    {
        f2Size[0] = 0;
        test_time(write_test3,f1,f2,f2Size);
    }
    printf("%d,%d\n",f2Size[0],cT);

    printf("test3\n");
    for(int i=0;i<1;i++)
    {
        f2Size[0] = 0;
        test_time(write_test2,f1,f2,f2Size);
    }
    printf("%d,%d\n",f2Size[0],cT);

    printf("test4\n");
    for(int i=0;i<1;i++)
    {
        f2Size[0] = 0;
        test_time(write_test1,f1,f2,f2Size);
    }

    __CUDA_ERROR("");

//  for(int i=0;i<f2Size[0];i++)
//     {
//         printf("%d ",f2[i]);
//     }


    printf("%d,%d\n",f2Size[0],cT);
    std::cout << "Success!" << std::endl;
    return 0;
}