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
        int level,
        unsigned long long& time1,
        unsigned long long& time2,
        unsigned mask)
    {
        time1 = clock64();
        int newWeight = sourceWeight + destWeight;
        __syncwarp(mask);
        time2 = clock64();
        time1 = time2-time1;
        int2 toWrite = make_int2(level, newWeight);
        unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[destNode]),
                                        reinterpret_cast<unsigned long long &>(toWrite));
        __syncwarp(mask);
        time2 = clock64()-time2;
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
        int& upFounds,int& downFounds,
        unsigned long long &time1,
        unsigned long long &time2,
        unsigned long long &time3,
        unsigned long long &time4
        )
    {
        unsigned mask = tile.ballot(k<end);
        unsigned long long t1,t2,t3,t4;
        if(k<end)
        {
            t1 = clock64();
            int2 dest = devEdges[k];
            __syncwarp(mask);
            t1 = clock64()-t1;
            hdist_t newNodeWeight;
            int flag = Relax(nodeWeight.y,dest.x,dest.y,devDistances,newNodeWeight,level,t2,t3,mask);
            t4 = clock64();
            if(flag)
            {
                if(newNodeWeight.x == 0 && level>newNodeWeight.x)
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
//                 if(newNodeWeight.x <= validLevel) //TODO -- chl nodeInfo.d>0
//                 {
// #if UPDOWNREM
//                     downQueue[downFounds++] = dest.x | 0x80000000; 
// #else
//                     downQueue[downFounds++] = dest.x;
// #endif
//                 }
            }
            __syncwarp(mask);
            t4 = clock64()-t4;
            time1 +=t1;
            time2 +=t2;
            time3 +=t3;
            time4 +=t4;

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
        unsigned long long* times
// #ifdef DELTA
//         ,
//         int* bucket,

// #endif
        )
    {
        unsigned long long t1,t2,t3,t4,t5,t6,t7;
        t1 = t2 = t3 = t4 = t5 = 0;
        // int allEdge = 0,allwrite=0;
        int upQueue[REGLIMIT];
        int downQueue[REGLIMIT];
        int upFounds = 0 ,downFounds = 0;
        t7 = clock64();
        thread_block g = this_thread_block();
        thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
        // if(tile.thread_rank()==tile.size()-1)
        //     startTime = clock64();
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
            // allEdge += ((end-start)/16+1)*16;
            tile.sync();
            for(int k = start+tile.thread_rank();k<end+tile.thread_rank();k+=stride){
                upDealEdges<VW_SIZE>(tile,devNodes,devEdges,k,end,nodeWeight,devDistances,level,\
                validLevel,upQueue,downQueue,upFounds,downFounds,t1,t2,t3,t4);
                if(tile.any(upFounds>=REGLIMIT)){
                    t6 = clock64();
                    VWWrite<VW_SIZE,int>(tile,pDevF2Size,devF2,upFounds,upQueue);
                    tile.sync();
                    t6 = clock64()-t6;
                    t5+=t6;
                    // if(tile.thread_rank()==tile.size()-1)
                    //     allwrite+=(www/32+1)*32;
                    upFounds = 0;
                }
                // if(tile.any(downFounds>=REGLIMIT)){
                //     int www = VWWrite<VW_SIZE,int>(tile,pDevF3Size,devF3,downFounds,downQueue);
                //     // if(tile.thread_rank()==tile.size()-1)
                //     //     allwrite+=(www/32+1)*32;
                //     downFounds = 0;
                // }
            }
        }
        // devPrintfInt(32,upFounds,"");
        t6 = clock64();
        VWWrite<VW_SIZE,int>(tile,pDevF2Size,devF2,upFounds,upQueue);
        t6 = clock64()-t6;
                    t5+=t6;
        t7 = clock64()-t7;
        if(tile.thread_rank()==0)
        {
            times[VirtualID] = t1;
            times[VirtualID+IDStride] = t2;
            times[VirtualID+IDStride*2] = t3;
            times[VirtualID+IDStride*3] = t4;
            times[VirtualID+IDStride*4] = t5;
            times[VirtualID+IDStride*5] = t6;
        }
        // allwrite+=(www/32+1)*32;
        // www = VWWrite<VW_SIZE,int>(tile,pDevF3Size,devF3,downFounds,downQueue);
        // allwrite+=(www/32+1)*32;
        // if(tile.thread_rank()==tile.size()-1)
        // {
        //     endTime = clock64();
        //     times[VirtualID] = allEdge*2 +allwrite;
        // }
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
            int flag = 0;
            // int flag = Relax(nodeWeight.y,dest.x,dest.y,devDistances,newNodeWeight,level);
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

    template <int VW_SIZE>
	__global__ void GNRTest1(
        int4* __restrict__  devNodes,
		int2* __restrict__  devEdges,
		hdist_t*  devDistances,
		int* __restrict__  devF1,
        int* devF2,
        const int devF1Size,
        int bbb)
    {
        int Queue[REGLIMIT];
        devF1+=bbb;
        int Founds = 0;
        int level = 0;
        thread_block g = this_thread_block();
        thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
        const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / VW_SIZE;
        const int IDStride = gridDim.x * (BLOCKDIM / VW_SIZE);
        tile.sync(); 
                    int min = INT_MAX;       
        // devPrintfInt(1,devF1Size,"devF1Size");
        for (int i = VirtualID; i < devF1Size; i += IDStride){
            int start,end,stride;
            int index = devF1[i];
            upGetNodeInfo<VW_SIZE>(devNodes,index,start,end,stride);
            // int2 nodeWeight = devDistances[i];
            tile.sync();
            for(int k = start+tile.thread_rank();k<end+tile.thread_rank();k+=stride){
                if(k<end)
                {
                    int2 dest = devEdges[k];
                    int newWeight = dest.x + dest.y;
                    int2 toWrite;
                    toWrite.x = dest.x;
                    toWrite.y = newWeight;
                    // atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]), reinterpret_cast<unsigned long long &>(toWrite));
                    // devDistances[dest.x] = toWrite;
                    // atomicMin(&devF2[dest.x],newWeight);
                    devF2[dest.x] = newWeight;
                    // cub::ThreadStore<cub::STORE_CG>((int2*)devF2+dest.x,toWrite);
                    // devDistances[dest.x].y = newWeight;
                    // devF2[32*i+k] = toWrite;
                    // devPrintfInt(32,min,"");
                }
                tile.sync();
            }
        }
    }

    template <int VW_SIZE>
	__global__ void GNRTest2(
        int4* __restrict__  devNodes,
		int2* __restrict__  devEdges,
		hdist_t*  devDistances,
		int* __restrict__  devF1,
        int* devF2,
        const int devF1Size,
        int bbb)
    {
        int Queue[REGLIMIT];
        int Founds = 0;
        int level = 0;
         devF1+=bbb;
        thread_block g = this_thread_block();
        thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
        const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / VW_SIZE;
        const int IDStride = gridDim.x * (BLOCKDIM / VW_SIZE);
        tile.sync(); 
                    int min = INT_MAX;       
        // devPrintfInt(1,devF1Size,"devF1Size");
        for (int i = VirtualID; i < devF1Size; i += IDStride){
            int start,end,stride;
            int index = devF1[i];
            upGetNodeInfo<VW_SIZE>(devNodes,index,start,end,stride);
            // int2 nodeWeight = devDistances[i];
            tile.sync();
            for(int k = start+tile.thread_rank();k<end+tile.thread_rank();k+=stride){
                if(k<end)
                {
                    int2 dest = devEdges[k];
                    int newWeight = dest.x + dest.y;
                    int2 toWrite;
                    toWrite.x = dest.x;
                    toWrite.y = newWeight;
                    // atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]), reinterpret_cast<unsigned long long &>(toWrite));
                    // devDistances[dest.x] = toWrite;
                    // atomicMin(&devF2[dest.x],newWeight);
                    devF2[dest.x*2] = newWeight;
                    // cub::ThreadStore<cub::STORE_CG>((int2*)devF2+dest.x,toWrite);
                    // devDistances[dest.x].y = newWeight;
                    // devF2[32*i+k] = toWrite;
                    // devPrintfInt(32,min,"");
                }
                tile.sync();
            }
        }
    }
}