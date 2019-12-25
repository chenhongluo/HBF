#include <Device/PreDeal_KernelFunc.cuh>
#include <Device/GNR_KernelFunc.cuh>
#include <Device/cudaOverlayGraph.cuh>

#include <../config.cuh>
#include <vector>
#include <XLib.hpp>
#include <ctime>

using std::vector;

using namespace PreDeal;
using namespace cuda_graph;
using namespace graph;
using namespace Kernels;

void PreDealGraph(GraphPreDeal &g)
{
    cudaGraphPreDeal cg(g);
    int supLevel = 0;
    EdgeType iSdirected = g.gw.Direction;
    while (supLevel < ALL_LEVELS - 1)
    {
        cg.copyOut();
        if (iSdirected == EdgeType::DIRECTED)
        {
            cg.copyIn();
        }
        cg.copyCandidate();
        cg.copyOrders();
        hostPrintfSmall("edgeDiffrence");
        EdgeDifference<DEFAULT_VW>
            <<<GRIDDIM, BLOCKDIM, SMem_Per_Block<char, BLOCKDIM>::value>>>(cg.devOutNodes, cg.devOutEdges, cg.devOrders,
                                                                           cg.devCandidates, cg.devCandidateSize,
                                                                           cg.devOutLoads, cg.devOutDones);
        if (iSdirected == EdgeType::DIRECTED)
        {
            EdgeDifference<DEFAULT_VW>
                <<<GRIDDIM, BLOCKDIM, SMem_Per_Block<char, BLOCKDIM>::value>>>(cg.devInNodes, cg.devInEdges, cg.devOrders,
                                                                               cg.devCandidates, cg.devCandidateSize,
                                                                               cg.devInLoads, cg.devInDones);
        }
        hostPrintfSmall("NodeMarksCompute");
        NodeMarksCompute<<<GRIDDIM, BLOCKDIM, SMem_Per_Block<char, BLOCKDIM>::value>>>(cg.devCandidates, cg.devCandidateSize,
                                                                                       cg.devOutLoads, cg.devOutDones,
                                                                                       cg.devInLoads, cg.devInDones,
                                                                                       cg.devMarks);
        __CUDA_ERROR("NodeMarksCompute Kernel");

        cg.copyMarks();
        //printVector(g.Marks,20);
        hostPrintfSmall("SelectNodesThresholdOrTOP");
        SelectNodesThresholdOrTOP(g.Candidates, g.Marks,
                                  g.Orders, g.ContractedNodes, supLevel, selectNodesNum[supLevel], selectNodesStrategy);

        hostPrintfSmall("ShortCut");
        g.AddEdges.clear();
        ShortCutCPU(g.ContractedNodes, g.Orders, g.AddEdges, g.OutNodesVec, g.OutEdgesVec, g.InNodesVec, g.InEdgesVec);
        hostPrintfSmall("CSRMerge");
        hostPrintfSmall(g.AddEdges.size(), "newSize: ");
        CSRMerge(g.AddEdges, g.OutNodesVec, g.OutEdgesVec, 0);
        if (iSdirected == EdgeType::DIRECTED)
        {
            CSRMerge(g.AddEdges, g.InNodesVec, g.InEdgesVec, 1);
        }
        supLevel++;
    }
}
namespace cuda_graph
{
void cudaGNRGraph::GNRSearchMain(int source)
{
    cudaClear();
    int f1Size = cudaInit(source), f2Size, f3Size, f4Size;
    f2Size = f3Size = f4Size = 0;
    int zero = 0;
    int maxCB = (V % DistanceSplitSize == 0) ? (V / DistanceSplitSize) : (V / DistanceSplitSize + 1);
    // __CUDA_ERROR("GNRSearchMain Kernel");

#if LOGFRONTIER
    //LogHostPrintSmall("NEXT");
#endif
    // printfInt(i,"bucketIndex");
    int *devF1 = upF1;
    int *devF1Size = upF1Size;
    int *devF2 = upF2;
    int *devF2Size = upF2Size;
    int *devF3 = downF1;
    int *devF3Size = downF1Size;
    int *devF4 = downF2;
    int *devF4Size = downF2Size;
    int level = 1;
    int iter = 0;
    while (1)
    {
        // __CUDA_ERROR("GNRSearchMain Kernel");
#if LOGFRONTIER
        if (level < 0)
        {
            LogPrintfFrotier(devF1, devF1Size, "up_devF1");
            LogPrintfFrotier(devF3, devF3Size, "up_devF3");
        }
#endif
        iter++;
        printf("%d:%d\n", iter, f1Size);
        // hostPrintfSmall(f1Size, "f1Size: ");
        // hostPrintfSmall(f3Size, "f3Size: ");
        if (f1Size == 0)
        {
            level++;
            break;
        }
#if DISTANCESPLIT
        if (f1Size < DSNodes)
        {
            GNRSearchNorm<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devUpOutNodes, devUpOutEdges, devDistances, devF1, devF2, f1Size, devF2Size, level);
            __CUDA_ERROR("GNRSearchMain Kernel");
        }
        else
        {
            cudaClearBucket();
            // BucketDebug();
            GNRSearchForWrite<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devUpOutNodes, devUpOutEdges, devDistances, devF1, f1Size);
            // copyIntFrom(f1Size, devF1Size);
            // copyIntFrom(f2Size, devF2Size);
            // hostPrintfSmall(f1Size, "f1Size: ");
            // hostPrintfSmall(f2Size, "f2Size: ");
            // BucketDebug();
            // __CUDA_ERROR("GNRSearchMain Kernel");
            for (int kb = 0; kb < maxCB; kb += 1)
            {
                CoalescedWrite<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devDistances, kb, devF2, devF2Size, level);
                // __CUDA_ERROR("GNRSearchMain Kernel");
            }
            // copyIntFrom(f1Size, devF1Size);
            // copyIntFrom(f2Size, devF2Size);
        }
#else
        GNRSearchNorm<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devUpOutNodes, devUpOutEdges, devDistances, devF1, devF2, f1Size, devF2Size, level);
#endif
        std::swap<int *>(devF1, devF2);
        std::swap<int *>(devF1Size, devF2Size);
        copyIntFrom(f1Size, devF1Size);
        copyIntTo(zero, devF2Size);
        level++;
    }
    if (ALL_LEVELS != 0)
    {
#if DISTANCESPLIT
        if (V < DSNodes)
        {
            GNRSearchDownFirstTime<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devDownOutNodes, devDownOutEdges, devDistances, V, devF3, devF3Size, level);
        }
        else
        {
            cudaClearBucket();
            GNRSearchDownFirstTimeForWrite<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devDownOutNodes, devDownOutEdges, devDistances, V, level);
            for (int kb = 0; kb < maxCB; kb += 1)
                CoalescedWrite<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devDistances, kb, devF3, devF3Size, level);
        }
#else
        GNRSearchDownFirstTime<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devDownOutNodes, devDownOutEdges, devDistances, V, devF3, devF3Size, level);
#endif
        level++;
        copyIntFrom(f3Size, devF3Size);
        while (1)
        {
            if (f3Size == 0)
            {
                level++;
                break;
            }
#if DISTANCESPLIT
            if (f1Size < DSNodes)
            {
                GNRSearchNorm<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devDownOutNodes, devDownOutEdges, devDistances, devF3, devF4, f3Size, devF4Size, level);
            }
            else
            {
                cudaClearBucket();
                GNRSearchForWrite<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devDownOutNodes, devDownOutEdges, devDistances, devF3, f3Size);
                for (int kb = 0; kb < maxCB; kb += 1)
                    CoalescedWrite<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devDistances, kb, devF4, devF4Size, level);
            }
#else
            GNRSearchNorm<DEFAULT_VW><<<GRIDDIM, BLOCKDIM>>>(devDownOutNodes, devDownOutEdges, devDistances, devF3, devF4, f3Size, devF4Size, level);
#endif
            std::swap<int *>(devF3, devF4);
            std::swap<int *>(devF3Size, devF4Size);
            copyIntFrom(f3Size, devF3Size);
            copyIntTo(zero, devF4Size);
            level++;
        }
        __CUDA_ERROR("GNRSearchMain Kernel");
    }
    // printf("iter times:%d", iter);
}
} // namespace cuda_graph
