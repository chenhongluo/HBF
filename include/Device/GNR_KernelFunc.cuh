#include <cub/cub.cuh>
#include <XLib.hpp>
// #include <climts>
#include <../config.cuh>
#include "LOG.cuh"
#include "Write_KenelFunc.cuh"
#include "Get_Node_Info.cuh"

// #define GetNodeInfo upGetNodeInfo
// #include "Get_Node_Info.cuh"
// #undef GetNodeInfo

// #define NODESPLIT false
// #define EDGECOMPRESS false
// #define GetNodeInfo downGetNodeInfo
// #include "Get_Node_Info.cuh"
// #undef GetNodeInfo

namespace Kernels
{

    __device__ __forceinline__ 
    int Relax(
        int sourceWeight,
        int destNode,
        int destWeight,
        hdist_t* __restrict__ devDistances,
        hdist_t& newNodeWeight,
        int level)
    {
        int newWeight = sourceWeight + destWeight;
        int2 toWrite = make_int2(level, newWeight);
        unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[destNode]),
                                        reinterpret_cast<unsigned long long &>(toWrite));
        newNodeWeight = reinterpret_cast<int2 &>(aa);
        return newNodeWeight.y > newWeight;
    }


    template <int VW_SIZE>
    __device__ __forceinline__ void upDealEdges(
        thread_block_tile<VW_SIZE>& tile,
        int4* __restrict__  devNodes,
        int2* __restrict__  devEdges,
        int k,int end,
        hdist_t& nodeWeight,hdist_t* __restrict__ devDistances,
        int level,int validLevel,
        int* upQueue,int* downQueue,
        int& upFounds,int& downFounds)
    {
        // unsigned mask = tile.ballot(k<end);
        if(k<end)
        {
            int2 dest = devEdges[k];
            hdist_t newNodeWeight;
            int flag = Relax(nodeWeight.y,dest.x,dest.y,devDistances,newNodeWeight,level);
            if(flag)
            {
                if(level>newNodeWeight.x)
                {                 
#if NODESPLIT
                    for(int j= 0;j<devNodes[dest.x].z;j++)
                    {
                        upQueue[upFounds++] = (unsigned)dest.x | (j<<SplitBits);                       
                    }
#else
                    upQueue[upFounds++] = dest.x;
#endif
                }
                if(newNodeWeight.x <= validLevel) //TODO -- chl nodeInfo.d>0
                {
#if UPDOWNREM
                    downQueue[downFounds++] = dest.x | 0x80000000; 
#else
                    downQueue[downFounds++] = dest.x;
#endif
                }
            }
            // __syncwarp(mask);
        }
        // tile.sync();
    }

    template <int VW_SIZE>
	__global__ void GNRSearchUp(
        int4* __restrict__  devNodes,
		int2* __restrict__  devEdges,
		hdist_t* __restrict__ devDistances,
		int* __restrict__  devF1,
		int* __restrict__  devF2, //up
        int* __restrict__ devF3,  //down
        const int devF1Size,
        int* __restrict__ pDevF2Size,
        int* __restrict__ pDevF3Size,
        int level,
        int validLevel,
        int threshold,
        int* times
// #ifdef DELTA
//         ,
//         int* bucket,

// #endif
        )
    {
        int startTime,endTime;
        int upQueue[REGLIMIT];
        int downQueue[REGLIMIT];
        int upFounds = 0 ,downFounds = 0;
        thread_block g = this_thread_block();
        thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
        if(tile.thread_rank()==0)
            startTime = clock64();
        const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / VW_SIZE;
        const int IDStride = gridDim.x * (BLOCKDIM / VW_SIZE);
        tile.sync();        
        // devPrintfInt(1,devF1Size,"devF1Size");
        for (int i = VirtualID; i < devF1Size; i += IDStride){
            int start,end,stride;
            int index = devF1[i]; 
#if NODESPLIT
            index = (unsigned)index & SplitMask;
#endif  
            hdist_t nodeWeight = devDistances[index];
#ifdef DELTA

#endif
#if NODESPLIT
            int splits = (unsigned)index >> SplitBits;
            upGetNodeInfo<VW_SIZE>(devNodes,index,splits,start,end,stride);
#else
            upGetNodeInfo<VW_SIZE>(devNodes,index,start,end,stride);
#endif         
            
            tile.sync();
            for(int k = start+tile.thread_rank();k<end+tile.thread_rank();k+=stride){
                // devPrintfInt(128,k,"k");
                upDealEdges<VW_SIZE>(tile,devNodes,devEdges,k,end,nodeWeight,devDistances,level,\
                validLevel,upQueue,downQueue,upFounds,downFounds);
                if(tile.any(upFounds>=(REGLIMIT-MAXSplits))){
                    VWWrite<VW_SIZE,int>(tile,pDevF2Size,devF2,upFounds,upQueue);
                    upFounds = 0;
                }
                if(tile.any(downFounds>=REGLIMIT)){
                    VWWrite<VW_SIZE,int>(tile,pDevF3Size,devF3,downFounds,downQueue);
                    downFounds = 0;
                }
            }
        }
        VWWrite<VW_SIZE,int>(tile,pDevF2Size,devF2,upFounds,upQueue);
        VWWrite<VW_SIZE,int>(tile,pDevF3Size,devF3,downFounds,downQueue);
        if(tile.thread_rank()==0)
        {
            endTime = clock64();
            times[VirtualID] = endTime - startTime;
        }
    }

        template <int VW_SIZE>
    __device__ __forceinline__ void downDealEdges(
        thread_block_tile<VW_SIZE>& tile,
        int4* __restrict__  devNodes,
        int2* __restrict__  devEdges,
        int k,
        int end,
        hdist_t nodeWeight,
        hdist_t*  __restrict__ devDistances,
        int level,
        int* Queue,
        int& Founds)
    {
        // unsigned mask = tile.ballot(k<end);
        if(k<end)
        {
            int2 dest = devEdges[k];
            hdist_t newNodeWeight;
            int flag = Relax(nodeWeight.y,dest.x,dest.y,devDistances,newNodeWeight,level);
            if(flag)
            {
                if(level>newNodeWeight.x)
                {
                    Queue[Founds++] = dest.x;
                }
            }
            // __syncwarp(mask);
        }
        // tile.sync();
    }

    template <int VW_SIZE>
	__global__ void GNRSearchDown(
        int4* __restrict__  devNodes,
		int2* __restrict__  devEdges,
		hdist_t* __restrict__ devDistances,
		int* __restrict__  devF1,
		int* __restrict__  devF2,
        const int devF1Size,
        int* pDevF2Size,
        int level)
    {
        int Queue[REGLIMIT];
        int Founds = 0;
        thread_block g = this_thread_block();
        thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
        const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / VW_SIZE;
        const int IDStride = gridDim.x * (BLOCKDIM / VW_SIZE);
        tile.sync();        
        // devPrintfInt(1,devF1Size,"devF1Size");
        for (int i = VirtualID; i < devF1Size; i += IDStride){
            int start,end,stride;
            int index = devF1[i];
            downGetNodeInfo<VW_SIZE>(devNodes,index,start,end,stride);
            hdist_t nodeWeight = devDistances[index];
            tile.sync();
            for(int k = start+tile.thread_rank();k<end+tile.thread_rank();k+=stride){
                downDealEdges<VW_SIZE>(tile,devNodes,devEdges,k,end,nodeWeight,devDistances,level,Queue,Founds);
                if(tile.any(Founds>=REGLIMIT)){
                    VWWrite<VW_SIZE,int>(tile,pDevF2Size,devF2,Founds,Queue);
                    Founds = 0;
                }
            }
        }
        VWWrite<VW_SIZE,int>(tile,pDevF2Size,devF2,Founds,Queue);
    }
}