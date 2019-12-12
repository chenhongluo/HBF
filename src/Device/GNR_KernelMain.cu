#include <Device/PreDeal_KernelFunc.cuh>
#include <Device/GNR_KernelFunc.cuh>
#include <Device/cudaOverlayGraph.cuh>

#include <../config.cuh>
#include <vector>
#include <XLib.hpp>
#include<ctime>

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
        hostPrintfSmall("edgeDiffrence");
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
        hostPrintfSmall("NodeMarksCompute");
        NodeMarksCompute
        <<< GRIDDIM,BLOCKDIM,SMem_Per_Block<char, BLOCKDIM>::value >>>
        (cg.devCandidates,cg.devCandidateSize,
        cg.devOutLoads,cg.devOutDones,
        cg.devInLoads,cg.devInDones,
        cg.devMarks);
        __CUDA_ERROR("NodeMarksCompute Kernel");

        cg.copyMarks();
        //printVector(g.Marks,20);
        hostPrintfSmall("SelectNodesThresholdOrTOP");
        SelectNodesThresholdOrTOP(g.Candidates,g.Marks,
        g.Orders,g.ContractedNodes,supLevel,selectNodesNum[supLevel],selectNodesStrategy);

        hostPrintfSmall("ShortCut");
        g.AddEdges.clear();
        ShortCutCPU(g.ContractedNodes,g.Orders,g.AddEdges,g.OutNodesVec,g.OutEdgesVec,g.InNodesVec,g.InEdgesVec);
        hostPrintfSmall("CSRMerge");
        hostPrintfSmall(g.AddEdges.size(),"newSize: ");                                                            
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
    int f1Size = cudaInit(source),f2Size,f3Size = 1,f4Size;
    __CUDA_ERROR("GNRSearchMain Kernel");

#if LOGFRONTIER
    //LogHostPrintSmall("NEXT");
#endif
    // printfInt(i,"bucketIndex");
    int* devF1 = upF1;
    int* devF1Size = upF1Size;
    int* devF2 = upF2;
    int* devF2Size = upF2Size;
    int* devF3 = downF1;
    int* devF3Size = downF1Size;
    int* devF4 = downF2;
    int* devF4Size = downF2Size;
    int level = 1;
    int validLevel = 1;
    int upFlag = 0,downFlag = 0,zero = 0;
    int* times;
    cudaMallocManaged(&times, GRIDDIM*BLOCKDIM/32);
    while(1)
    {
        clock_t time1 = clock();
        for(int i=0;i<GRIDDIM*BLOCKDIM/32;i++)
            times[i]=0;
        // printStr("GNRSearchUp");
#if LOGFRONTIER
    if(level<0){
        LogPrintfFrotier(devF1,devF1Size,"up_devF1");
        LogPrintfFrotier(devF3,devF3Size,"up_devF3");
    }
#endif
        // hostPrintfSmall(f1Size,"f1Size: ");
        // hostPrintfSmall(f3Size,"f3Size: ");
        if(f1Size<50000&&f3Size==0){
        //if(f1Size==0){
            break;
        }
        upFlag = downFlag = 0;
        if(f1Size<KERNELNODES && f3Size > 0)
            downFlag = 1;
        else
            downFlag = 0;
        if(f1Size>0){
            int gridDim1 = (f1Size / 4+1)<GRIDDIM?(f1Size /4+1):GRIDDIM;
            if(downFlag)
                GNRSearchUp<DEFAULT_VW><<<gridDim1,BLOCKDIM,0,upStream>>>
                (devUpOutNodes,devUpOutEdges,devDistances,
                devF1,devF2,devF4,f1Size,devF2Size,devF4Size,
                level,validLevel,0,times);
            else
                GNRSearchUp<DEFAULT_VW><<<GRIDDIM,BLOCKDIM,0,upStream>>>
                (devUpOutNodes,devUpOutEdges,devDistances,
                devF1,devF2,devF3,f1Size,devF2Size,devF3Size,
                level,validLevel,0,times);
            std::swap<int*>(devF1,devF2);
            std::swap<int*>(devF1Size,devF2Size);
            // copyIntFromAsync(f1Size,devF2Size,upStream);
            upFlag = 1;
        }
        if(downFlag == 1)
        {
            GNRSearchDown<DEFAULT_VW><<<GRIDDIM,BLOCKDIM,0,downStream>>>
            (devDownOutNodes,devDownOutEdges,devDistances,
            devF3,devF4,f3Size,devF4Size,
            level);
            validLevel = level;
            std::swap<int*>(devF3,devF4);
            std::swap<int*>(devF3Size,devF4Size);
        }
        if(upFlag == 1)
            cudaStreamSynchronize(upStream);
        if(downFlag == 1)
            cudaStreamSynchronize(downStream);
        if(upFlag == 1){
            clock_t time2 = clock();
            copyIntFrom(f1Size,devF1Size);
            copyIntTo(zero,devF2Size);
            if(!downFlag)
                copyIntFrom(f3Size,devF3Size);
            cout<<f1Size<<" "<<time2-time1<<endl;
            for(int i=0;i<GRIDDIM*BLOCKDIM/32;i++)
                cout<<times[i]<<" ";
            cout<<endl;
        }
        if(downFlag == 1)
        {
            copyIntFrom(f3Size,devF3Size);
            copyIntTo(zero,devF4Size);
        }
        __CUDA_ERROR("GNRSearchMain Kernel");
        level++;
    }
}
}
