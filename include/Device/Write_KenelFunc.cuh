#include <cub/cub.cuh>
#include <XLib.hpp>
#include <climits>
#include <../config.cuh>
#include <cooperative_groups.h>

using namespace cooperative_groups;

namespace Kernels
{
    template<int VW_SIZE,typename T>
    __device__ __forceinline__
    void VWWrite(thread_block_tile<VW_SIZE>& tile,int *pAllsize,T* writeStartAddr,
            const int& writeCount,T* data)
    {
        int sum = 0;
        int bias = 0;
        VWInclusiveScanAdd<VW_SIZE,int>(tile,writeCount,sum);
        // devPrintfInt(32,sum,"sum");
        if(tile.thread_rank() == tile.size()-1 && sum !=0)
        {
            bias = atomicAdd(pAllsize,sum);
        }
        bias = tile.shfl(bias,tile.size()-1);
        sum -= writeCount;
        
        for(int it = 0;it<writeCount;it++)
        {
            cub::ThreadStore<cub::STORE_CG>(writeStartAddr+bias+sum+it,data[it]);           
        }
    }

    template<int VW_SIZE,typename T>
    __device__ __forceinline__
    void VWWriteForSplits(thread_block_tile<VW_SIZE>& tile,int *pAllsize,T* writeStartAddr,
            const int& writeCount,T* data,int* splitsNum,int len)
    {
        int sum = 0;
        int bias = 0;
        // devPrintfInt(32,writeCount,"writeCount");
        // devPrintfInt(32,len,"len");
        VWInclusiveScanAdd<VW_SIZE,int>(tile,writeCount,sum);
        if(tile.thread_rank() == tile.size()-1 && sum !=0)
        {
            bias = atomicAdd(pAllsize,sum);
        }
        bias = tile.shfl(bias,tile.size()-1);
        sum -= writeCount;
        int temp = 0;
        for(int i = 0;i<len;i++)
        {
            int node = data[i];
            int s = splitsNum[i];
            for(int j= 0;j<s;j++)
            {
                node = (unsigned)node | (j<<SplitBits);
                cub::ThreadStore<cub::STORE_CG>(writeStartAddr+bias+sum+temp+j,node); 
            }
            temp += s;      
        }
    }
}