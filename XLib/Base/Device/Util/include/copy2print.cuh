#pragma once

#include <iostream>
#include <string>
#include <limits>
#include <vector_types.h>
#include <vector_functions.hpp>
#include <vector>
#include <../config.cuh>
using std::vector;
#if __NVCC__
template<typename T>
inline void copyVecTo(vector<T>& host,T*& dev)
{
    cudaMemcpy(dev, &(host[0]), (host.size()) * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void copyVecFrom(vector<T>& host,T*& dev)
{
    cudaMemcpy(&(host[0]), dev, (host.size()) * sizeof(T), cudaMemcpyDeviceToHost);
}

inline void copyIntTo(int& host,int *dev)
{
    cudaMemcpy(dev, &host, 1 * sizeof(int),cudaMemcpyHostToDevice);
}

inline void copyIntFrom(int& host,int* dev)
{
    cudaMemcpy(&host, dev, 1 * sizeof(int),cudaMemcpyDeviceToHost);
}

template<typename T>
inline void copyVecToAsync(vector<T>& host,T*& dev,cudaStream_t s)
{
    cudaMemcpyAsync(dev, &(host[0]), (host.size()) * sizeof(T), cudaMemcpyHostToDevice,s);
}

template<typename T>
inline void copyVecFromAsync(vector<T>& host,T*& dev,cudaStream_t s)
{
    cudaMemcpyAsync(&(host[0]), dev, (host.size()) * sizeof(T), cudaMemcpyDeviceToHost,s);
}

inline void copyIntToAsync(int& host,int *dev,cudaStream_t s)
{
    cudaMemcpyAsync(dev, &host, 1 * sizeof(int),cudaMemcpyHostToDevice,s);
}

inline void copyIntFromAsync(int& host,int* dev,cudaStream_t s)
{
    cudaMemcpyAsync(&host, dev, 1 * sizeof(int),cudaMemcpyDeviceToHost,s);
}

__device__ __forceinline__ void devPrintfInt(int k,int n,char* s)
{
    if(blockIdx.x * BLOCKDIM + threadIdx.x < k)
    {
        printf("%s:%d,id:%d\n",s,n,blockIdx.x * BLOCKDIM + threadIdx.x);
    }
}

__device__ __forceinline__ void devPrintfX(int k,int n,char* s)
{
    if(blockIdx.x * BLOCKDIM + threadIdx.x < k)
    {
        printf("%s:%0x,id:%d\n",s,n,blockIdx.x * BLOCKDIM + threadIdx.x);
    }
}

__device__ __forceinline__ void devPrintfInt2(int k,int2 n,char* s)
{
    if(blockIdx.x * BLOCKDIM + threadIdx.x < k)
    {
        printf("%s:%d,%d,id:%d\n",s,n.x,n.y,blockIdx.x * BLOCKDIM + threadIdx.x);
    }
}

__device__ __forceinline__ void devPrintfInt4(int k,int4 n,char* s)
{
    if(blockIdx.x * BLOCKDIM + threadIdx.x < k)
    {
        printf("%s:%d,%d,%d,%d,id:%d\n",s,n.x,n.y,n.z,n.w,blockIdx.x * BLOCKDIM + threadIdx.x);
    }
}

__device__ __forceinline__ void devPrintfulong(int k,unsigned long long n,char* s)
{
    if(blockIdx.x * BLOCKDIM + threadIdx.x < k)
    {
        printf("%s:%lld,%d,id:%d\n",s,n,sizeof(n),blockIdx.x * BLOCKDIM + threadIdx.x);
    }
}


#endif