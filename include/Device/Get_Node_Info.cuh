#include <XLib.hpp>
#include <../config.cuh>
namespace Kernels
{
    template <int VW_SIZE>
    __device__ __forceinline__ 
    void upGetNodeInfo(
        int4* __restrict__ devNodes,
        int index,
#if NODESPLIT
        int splits,
#endif
        int& start,
        int& end,
        int& stride)
    {
#if EDGECOMPRESS
        int temp = COMPRESSSIZE;
#else
        int temp = VW_SIZE;
#endif
        int4 nodeInfo = devNodes[index];
        // devPrintfInt4(32,nodeInfo,"nodeInfo ");
        end = nodeInfo.y;
        //devPrintfInt4(32*32,nodeInfo,"");
#if NODESPLIT
        stride = nodeInfo.z * temp;
        start = nodeInfo.x + splits * temp;
#else
        stride = temp;
        start = nodeInfo.x;
#endif
    }

    template <int VW_SIZE>
    __device__ __forceinline__ 
    void downGetNodeInfo(
        int4* __restrict__ devNodes,
        int& index,
        int& start,
        int& end,
        int& stride)
    {  
        int temp = VW_SIZE;
#if UPDOWNREM
        index = (unsigned)index & 0x7FFFFFFF;
#endif
        int4 nodeInfo = devNodes[index];
#if UPDOWNREM
        if((unsigned)index >> 31){
            end = nodeInfo.z;
        }
        else{
            end = nodeInfo.y;
        }
#else
        end = nodeInfo.y;
#endif
        stride = temp;
        start = nodeInfo.x;
    }
}