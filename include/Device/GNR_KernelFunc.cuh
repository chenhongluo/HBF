#include <cub/cub.cuh>
#include <XLib.hpp>
// #include <climts>
#include <../config.cuh>
#include "LOG.cuh"
// #include "Write_KenelFunc.cuh"
// #include "Get_Node_Info.cuh"

// #define GetNodeInfo upGetNodeInfo
// #include "Get_Node_Info.cuh"
// #undef GetNodeInfo

// #define NODESPLIT false
// #define EDGECOMPRESS false
// #define GetNodeInfo downGetNodeInfo
// #include "Get_Node_Info.cuh"
// #undef GetNodeInfo

extern __constant__ int2 *buckets[];
extern __constant__ int *bucketSizes[];

namespace Kernels
{
template <int VW_SIZE>
__global__ void GNRSearchNorm(
    int4 *__restrict__ devNodes,
    int2 *__restrict__ devEdges,
    hdist_t *__restrict__ devDistances,
    int *__restrict__ devF1,
    int *__restrict__ devF2,
    const int devF1Size,
    int *__restrict__ pDevF2Size,
    int level)
{
    //alloc node&edge
    const int RealID = blockIdx.x * BLOCKDIM + threadIdx.x;
    const int tileID = RealID / VW_SIZE;
    const int tileRank = threadIdx.x % VW_SIZE;
    const int IDStride = gridDim.x * (BLOCKDIM / VW_SIZE);

    // for write in shared mem
    __shared__ int st[SHAREDLIMIT * BLOCKDIM / VW_SIZE];
    int *queue = st + threadIdx.x / VW_SIZE * SHAREDLIMIT;
    int Founds = 0;
    unsigned mymask = (1 << tileRank) - 1;
    int globalBias;

    //alloc node for warps
    for (int i = tileID; i < devF1Size; i += IDStride)
    {
        int index = devF1[i];
        int sourceWeight = devDistances[index].y;
        // devPrintf(1, sourceWeight, "sourceWeight");
        // devPrintf(128, tileRank, "tileRank");
        int4 nodeInfo = devNodes[index];
        // __syncwarp(0xFFFFFFFF);
        // alloc edges in a warp
        for (int k = nodeInfo.x + tileRank; k < nodeInfo.y + tileRank; k += VW_SIZE)
        {
            //relax edge  if flag=1 write to devF2
            int flag = 0;
            int2 dest;
            if (k < nodeInfo.y)
            {
                dest = devEdges[k];
                int newWeight = sourceWeight + dest.y;
                int2 toWrite = make_int2(level, newWeight);
                unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
                                                  reinterpret_cast<unsigned long long &>(toWrite));
                hdist_t &oldNode2Weight = reinterpret_cast<int2 &>(aa);
                flag = ((oldNode2Weight.y > newWeight) && (level > oldNode2Weight.x));
            }
            unsigned mask = __ballot_sync(0xFFFFFFFF, flag);
            // devPrintfX(32, mask, "mask");

            int sum = __popc(mask);
            if (sum + Founds > SHAREDLIMIT)
            {
                // write to global mem if larger than shared mem
                if (tileRank == 0)
                    globalBias = atomicAdd(pDevF2Size, Founds);
                globalBias = __shfl_sync(0xFFFFFFFF, globalBias, 0);
                for (int j = tileRank; j < Founds; j += 32)
                    devF2[globalBias + j] = queue[j];
                __syncwarp(0xFFFFFFFF);
                Founds = 0;
            }
            if (flag)
            {
                // write to shared mem
                mask = mask & mymask;
                int pos = __popc(mask);
                queue[pos + Founds] = dest.x;
            }
            __syncwarp(0xFFFFFFFF);
            Founds += sum;
        }
    }
    // write to global mem
    if (tileRank == 0)
        globalBias = atomicAdd(pDevF2Size, Founds);
    globalBias = __shfl_sync(0xFFFFFFFF, globalBias, 0);
    for (int j = tileRank; j < Founds; j += 32)
        devF2[globalBias + j] = queue[j];
}

template <int VW_SIZE>
__global__ void GNRSearchForWrite(
    int4 *__restrict__ devNodes,
    int2 *__restrict__ devEdges,
    hdist_t *__restrict__ devDistances,
    int *__restrict__ devF1,
    const int devF1Size)
{
    //alloc node&edge
    const int RealID = blockIdx.x * BLOCKDIM + threadIdx.x;
    const int tileID = RealID / VW_SIZE;
    const int tileRank = threadIdx.x % VW_SIZE;
    const int IDStride = gridDim.x * (BLOCKDIM / VW_SIZE);
    //alloc node for warps
    for (int i = tileID; i < devF1Size; i += IDStride)
    {
        int index = devF1[i];
        int sourceWeight = devDistances[index].y;
        int4 nodeInfo = devNodes[index];
        // __syncwarp(0xFFFFFFFF);
        // alloc edges in a warp
        for (int k = nodeInfo.x + tileRank; k < nodeInfo.y + tileRank; k += VW_SIZE)
        {
            int2 dest;
            if (k < nodeInfo.y)
            {
                dest = devEdges[k];
                // devPrintfInt2(32,dest,"dest");
                int newWeight = sourceWeight + dest.y;
                int2 toWrite = make_int2(dest.x, newWeight);
                // devPrintfInt2(32,toWrite,"toWrite");
                int kb = (dest.x / DistanceSplitSize);
                // devPrintfInt(32,kb,"kb");
                int pos = atomicAdd(bucketSizes[kb], 1);
                // devPrintfInt(32,pos,"pos");
                int2 *bb = buckets[kb];
                bb[pos] = toWrite;
            }
        }
    }
}

template <int VW_SIZE>
__global__ void CoalescedWrite(
    hdist_t *__restrict__ devDistances,
    int kb,
    int *__restrict__ devF2,
    int *__restrict__ pDevF2Size,
    int level)
{
    const int tileRank = threadIdx.x % VW_SIZE;

    // for write in shared mem
    __shared__ int st[SHAREDLIMIT * BLOCKDIM / VW_SIZE];
    int *queue = st + threadIdx.x / VW_SIZE * SHAREDLIMIT;
    int Founds = 0;
    unsigned mymask = (1 << tileRank) - 1;
    int globalBias;

    const int start = blockIdx.x * BLOCKDIM + threadIdx.x;
    int end = *(bucketSizes[kb]);
    // end = (end / 32 + 1) * 32;
    const int stride = gridDim.x * BLOCKDIM;
    int2 *data = buckets[kb];

    for (int k = start; k < end + start; k += stride)
    {
        int flag = 0;
        int2 index2distance;
        if (k < end)
        {
            index2distance = data[k];
            // devPrintfInt2(32, index2distance, "index2distance");
            // devPrintfInt2(32, devDistances[index2distance.x], "devDistances[index2distance.x]");
            int2 toWrite = make_int2(level, index2distance.y);
            unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[index2distance.x]),
                                              reinterpret_cast<unsigned long long &>(toWrite));
            hdist_t &oldNode2Weight = reinterpret_cast<int2 &>(aa);
            flag = ((oldNode2Weight.y > index2distance.y) && (level > oldNode2Weight.x));
            // devPrintfInt2(32, oldNode2Weight, "oldNode2Weight");
        }
        unsigned mask = __ballot_sync(0xFFFFFFFF, flag);
        // devPrintfX(32, mask, "mask");

        int sum = __popc(mask);
        // devPrintfX(32, sum, "mask");
        if (sum + Founds > SHAREDLIMIT)
        {
            // write to global mem if larger than shared mem
            if (tileRank == 0)
                globalBias = atomicAdd(pDevF2Size, Founds);
            globalBias = __shfl_sync(0xFFFFFFFF, globalBias, 0);
            for (int j = tileRank; j < Founds; j += 32)
                devF2[globalBias + j] = queue[j];
            __syncwarp(0xFFFFFFFF);
            Founds = 0;
        }
        if (flag)
        {
            // write to shared mem
            mask = mask & mymask;
            int pos = __popc(mask);
            queue[pos + Founds] = index2distance.x;
        }
        __syncwarp(0xFFFFFFFF);
        Founds += sum;
    }
    // write to global mem
    if (tileRank == 0)
        globalBias = atomicAdd(pDevF2Size, Founds);
    globalBias = __shfl_sync(0xFFFFFFFF, globalBias, 0);
    for (int j = tileRank; j < Founds; j += 32)
        devF2[globalBias + j] = queue[j];
    __syncwarp(0xFFFFFFFF);
}

template <int VW_SIZE>
__global__ void GNRSearchDownFirstTime(
    int4 *__restrict__ devNodes,
    int2 *__restrict__ devEdges,
    hdist_t *__restrict__ devDistances,
    int V,
    int *__restrict__ devF2,
    int *__restrict__ pDevF2Size,
    int level)
{
    //alloc node&edge
    const int RealID = blockIdx.x * BLOCKDIM + threadIdx.x;
    const int tileID = RealID / VW_SIZE;
    const int tileRank = threadIdx.x % VW_SIZE;
    const int IDStride = gridDim.x * (BLOCKDIM / VW_SIZE);

    // for write in shared mem
    __shared__ int st[SHAREDLIMIT * BLOCKDIM / VW_SIZE];
    int *queue = st + threadIdx.x / VW_SIZE * SHAREDLIMIT;
    int Founds = 0;
    unsigned mymask = (1 << tileRank) - 1;
    int globalBias;

    //alloc node for warps
    for (int i = tileID; i < V; i += IDStride)
    {
        int index = i;
        int sourceWeight = devDistances[index].y;
        if (sourceWeight == INT_MAX)
            continue;
        // devPrintf(1, sourceWeight, "sourceWeight");
        // devPrintf(128, tileRank, "tileRank");
        int4 nodeInfo = devNodes[index];
        // __syncwarp(0xFFFFFFFF);
        // alloc edges in a warp
        for (int k = nodeInfo.x + tileRank; k < nodeInfo.w + tileRank; k += VW_SIZE)
        {
            //relax edge  if flag=1 write to devF2
            int flag = 0;
            int2 dest;
            if (k < nodeInfo.w)
            {
                dest = devEdges[k];
                int newWeight = sourceWeight + dest.y;
                int2 toWrite = make_int2(level, newWeight);
                unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
                                                  reinterpret_cast<unsigned long long &>(toWrite));
                hdist_t &oldNode2Weight = reinterpret_cast<int2 &>(aa);
                flag = ((oldNode2Weight.y > newWeight) && (level > oldNode2Weight.x));
            }
            unsigned mask = __ballot_sync(0xFFFFFFFF, flag);
            // devPrintfX(32, mask, "mask");

            int sum = __popc(mask);
            if (sum + Founds > SHAREDLIMIT)
            {
                // write to global mem if larger than shared mem
                if (tileRank == 0)
                    globalBias = atomicAdd(pDevF2Size, Founds);
                globalBias = __shfl_sync(0xFFFFFFFF, globalBias, 0);
                for (int j = tileRank; j < Founds; j += 32)
                    devF2[globalBias + j] = queue[j];
                __syncwarp(0xFFFFFFFF);
                Founds = 0;
            }
            if (flag)
            {
                // write to shared mem
                mask = mask & mymask;
                int pos = __popc(mask);
                queue[pos + Founds] = dest.x;
            }
            __syncwarp(0xFFFFFFFF);
            Founds += sum;
        }
    }
    // write to global mem
    if (tileRank == 0)
        globalBias = atomicAdd(pDevF2Size, Founds);
    globalBias = __shfl_sync(0xFFFFFFFF, globalBias, 0);
    for (int j = tileRank; j < Founds; j += 32)
        devF2[globalBias + j] = queue[j];
}

template <int VW_SIZE>
__global__ void GNRSearchDownFirstTimeForWrite(
    int4 *__restrict__ devNodes,
    int2 *__restrict__ devEdges,
    hdist_t *__restrict__ devDistances,
    int V,
    int level)
{
    //alloc node&edge
    const int RealID = blockIdx.x * BLOCKDIM + threadIdx.x;
    const int tileID = RealID / VW_SIZE;
    const int tileRank = threadIdx.x % VW_SIZE;
    const int IDStride = gridDim.x * (BLOCKDIM / VW_SIZE);
    //alloc node for warps
    for (int i = tileID; i < V; i += IDStride)
    {
        int index = i;
        int sourceWeight = devDistances[index].y;
        if (sourceWeight == INT_MAX)
            continue;
        int4 nodeInfo = devNodes[index];
        // __syncwarp(0xFFFFFFFF);
        // alloc edges in a warp
        for (int k = nodeInfo.x + tileRank; k < nodeInfo.w + tileRank; k += VW_SIZE)
        {
            int2 dest;
            if (k < nodeInfo.w)
            {
                dest = devEdges[k];
                // devPrintfInt2(32,dest,"dest");
                int newWeight = sourceWeight + dest.y;
                int2 toWrite = make_int2(dest.x, newWeight);
                // devPrintfInt2(32,toWrite,"toWrite");
                int kb = (dest.x / DistanceSplitSize);
                // devPrintfInt(32,kb,"kb");
                int pos = atomicAdd(bucketSizes[kb], 1);
                // devPrintfInt(32,pos,"pos");
                (buckets[kb])[pos] = toWrite;
            }
        }
    }
}
} // namespace Kernels