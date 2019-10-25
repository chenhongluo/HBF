#include <Device/PreDeal_KernelFun.cuh>
#include <Device/GNR_KernelFun.cuh>
#include <Device/cudaOverlayGraph.cuh>

#include <../config.cuh>
#include <vector>
#include <XLib.hpp>

using std::vector;

using namespace PreDeal;
using namespace cuda_graph;
using namespace graph;
using namespace Kernels;

void PreDealGraph(GraphPreDeal& g)
{
    cudaGraphPreDeal cg(g);
    int supLevel = 0;
    EdgeType iSdirected = g.gw.Direction;
    while(supLevel<ALL_LEVELS-1)
    {
        cg.copyOut();
        if(iSdirected == EdgeType::DIRECTED)
        {
            cg.copyIn();
        }
        cg.copyCandidate();
        cg.copyOrders();
        printfStr("edgeDiffrence");
        EdgeDifference<DEFAULT_VW> 
        <<< GRIDDIM,BLOCKDIM,SMem_Per_Block<char, BLOCKDIM>::value>>>
        (cg.devOutNodes,cg.devOutEdges,cg.devOrders,
        cg.devCandidates,cg.devCandidateSize,
        cg.devOutLoads,cg.devOutDones);
        if(iSdirected == EdgeType::DIRECTED)
        {
            EdgeDifference<DEFAULT_VW> 
            <<< GRIDDIM,BLOCKDIM,SMem_Per_Block<char, BLOCKDIM>::value>>>
            (cg.devInNodes,cg.devInEdges,cg.devOrders,
            cg.devCandidates,cg.devCandidateSize,
            cg.devInLoads,cg.devInDones);
        }
        printfStr("NodeMarksCompute");
        NodeMarksCompute
        <<< GRIDDIM,BLOCKDIM,SMem_Per_Block<char, BLOCKDIM>::value >>>
        (cg.devCandidates,cg.devCandidateSize,
        cg.devOutLoads,cg.devOutDones,
        cg.devInLoads,cg.devInDones,
        cg.devMarks);
        __CUDA_ERROR("NodeMarksCompute Kernel");

        cg.copyMarks();
        //printVector(g.Marks,20);
        printf("SelectNodesThresholdOrTOP\n");
        SelectNodesThresholdOrTOP(g.Candidates,g.Marks,
        g.Orders,g.ContractedNodes,supLevel,selectNodesNum[supLevel],selectNodesStrategy);
        // __CUDA_ERROR("SelectNodesThresholdOrTOP Kernel");
        // g.Candidates.remove(29);
        // g.Candidates.remove(1048599);
        // g.Orders[29] = 0;
        // g.Orders[1048599] = 0;
        // g.ContractedNodes.push_back(29);
        // g.ContractedNodes.push_back(1048599);

        // printf("ContractedNodes:/n");
        // printVector(g.ContractedNodes,g.ContractedNodes.size());

        // printf("g.Orders:/n");
        // printVector<int>(g.Orders,g.Orders.size());


        // printf("GetTriEdges\n");
        // GetTriEdges(g.ContractedNodes,g.Orders,supLevel,g.InNodesVec,g.InEdgesVec,
        // g.ContractedEdges);
        // for(int i=0;i<g.ContractedEdges.size();i++)
        // {
        //     printf("x:%d,y:%d,z:%d\n",g.ContractedEdges[i].x,
        //     g.ContractedEdges[i].y,g.ContractedEdges[i].z);
        // }

        // for(int x:g.ContractedNodes)
        // {
        //     printf("%d\n",g.Orders[x]);
        // }

        printfStr("ShortCut");
        g.AddEdges.clear();
        ShortCutCPU(g.ContractedNodes,g.Orders,g.AddEdges,g.OutNodesVec,g.OutEdgesVec,g.InNodesVec,g.InEdgesVec);
        // cg.copyTriEdges();
        // cg.copyOrders();
        // int3* devF1 = cg.devTriEdges;
        // int* devF1Size = cg.devTriEdgeSize;
        // int3* devF2 = cg.devNewEdges;
        // int* devF2Size = cg.devNewEdgeSize;
        // cg.InitSize(cg.devNewEdgeSize);
        // cg.InitSize(cg.devNewEndEdgeSize);
        // while(1)
        // {
        //     int size,newSize;
        //     cg.printfEdges(devF2,devF2Size);
        //     ShortCut<DEFAULT_VW> 
        //     <<< GRIDDIM,BLOCKDIM,SMem_Per_Block<char, BLOCKDIM>::value>>>
        //     (devF1,devF1Size,
        //     cg.devOutNodes,cg.devOutEdges,cg.devOrders,
        //     devF2,devF2Size,
        //     cg.devNewEndEdges,cg.devNewEndEdgeSize);
        //     cg.printfEdges(devF1,devF1Size);
            // cg.printfEdges(devF2,devF2Size);

        //     //cg.copyAddEdges();
        //     cg.copyLeftEdgesSize(devF2Size,size);
        //     cg.copyLeftEdgesSize(cg.devNewEndEdgeSize,newSize);
        //     //TODO CPU?? chl
        //     printf("%d,%d\n",size,newSize);
        //     if(size == 0)
        //     {
        //         break;
        //     }

        //     std::swap<int3*>(devF1,devF2);
        //     std::swap<int*>(devF1Size,devF2Size);
        //     cg.InitSize(devF2Size);
        // }
        // cg.InitSize(cg.devNewEdgeSize);
        // cg.copyAddEdges();
        // printVector(g.AddEdges,g.AddEdges.size());
        printfStr("CSRMerge");
        printfInt(g.AddEdges.size(),"newSize");                                                            
        CSRMerge(g.AddEdges,g.OutNodesVec,g.OutEdgesVec,0);
        if(iSdirected == EdgeType::DIRECTED)
        {
            CSRMerge(g.AddEdges,g.InNodesVec,g.InEdgesVec,1);
        }
        supLevel++;
    }
}
namespace cuda_graph {
void cudaGNRGraph::GNRSearchMain(int source)
{
    cudaClear();
    int i = cudaInit(source);
    __CUDA_ERROR("BellmanFord Kernel");

#if LOGFRONTIER
    LogPrintfStr("NEXT");
#endif

    for (;i<cudaUpBuckets.size();i++)
    {
        // printfInt(i,"bucketIndex");
        int* devF1 = cudaUpBuckets[i].devF;
        int* devF1Size = cudaUpBuckets[i].devFSize;
        int* devF2 = devFt;
        int* devF2Size = devFtSize;
        int sublevel=1;
        int suplevel=i;
        while(1)
        {
            // printStr("GNRSearchUp");
#if LOGFRONTIER
            LogPrintfFrotier(devF1,devF1Size,"up_devF1");
#endif
            GNRSearchUp<DEFAULT_VW> 
            <<< GRIDDIM,BLOCKDIM,SMem_Per_Block<char, BLOCKDIM>::value>>>
            (devUpOrderTrans,devDownOrderTrans,
            devUpOutNodes,devUpOutEdges,
            devDistances,
            devF1,devF2,devF1Size,devF2Size,
            devUpBuckets,devUpBucketSize,
            devDownBuckets,devDownBucketSize,
            suplevel,sublevel);
            __CUDA_ERROR("BellmanFord Kernel");
            sublevel++;
            int test_size = GetSize(devF2Size);
            std::swap<int*>(devF1,devF2);
            std::swap<int*>(devF1Size,devF2Size);
            InitSize(devF2Size);
            if(test_size == 0)
            {
                break;
            }
        }
        InitSize(devFtSize);
    }
    InitSize(devFtSize);
    // printfBuckets();
    for (i=cudaDownBuckets.size()-1;i>=0;i--)
    {
        //printInt(i);       
        int* devF1 = cudaDownBuckets[i].devF;
        int* devF1Size = cudaDownBuckets[i].devFSize;
        int* devF2 = devFt;
        int* devF2Size = devFtSize;
        int sublevel=1;
        int suplevel=i;
        while(1)
        {
            //printStr("GNRSearchDown");
#if LOGFRONTIER
            LogPrintfFrotier(devF1,devF1Size,"down_devF1");
#endif
            GNRSearchDown<DEFAULT_VW> 
            <<< GRIDDIM,BLOCKDIM,SMem_Per_Block<char, BLOCKDIM>::value>>>
            (devUpOrderTrans,devDownOrderTrans,
            devDownOutNodes,devDownOutEdges,
            devDistances,
            devF1,devF2,devF1Size,devF2Size,
            devUpBuckets,devUpBucketSize,
            devDownBuckets,devDownBucketSize,
            suplevel,sublevel);
            __CUDA_ERROR("BellmanFord Kernel");
            sublevel++;
            int test_size = GetSize(devF2Size);
            std::swap<int*>(devF1,devF2);
            std::swap<int*>(devF1Size,devF2Size);
            InitSize(devF2Size);
            // printInt(test_size);
            if(test_size == 0)
            {
                break;
            }
        }
        InitSize(devFtSize);
    }
    InitSize(devFtSize);
}
}
