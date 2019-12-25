#include <XLib.hpp>
#include <../config.cuh>
namespace Kernels
{
template <int VW_SIZE>
__device__ __forceinline__ void upGetNodeInfo(
    int4 *__restrict__ devNodes,
    int &index,
    int &start,
    int &end,
    int &stride)
{
#if EDGECOMPRESS
        int temp = COMPRESSSIZE;
#else
        int temp = VW_SIZE;
#endif
        int4 nodeInfo = devNodes[index];
        start = nodeInfo.x;
        end = nodeInfo.y;
        stride = temp;
}

template <int VW_SIZE>
__device__ __forceinline__ void downGetNodeInfo(
    int4 *__restrict__ devNodes,
    int &index,
    int &start,
    int &end,
    int &stride)
{
        int temp = VW_SIZE;

        int4 nodeInfo = devNodes[index];
        end = nodeInfo.y;
        stride = temp;
        start = nodeInfo.x;
}
} // namespace Kernels