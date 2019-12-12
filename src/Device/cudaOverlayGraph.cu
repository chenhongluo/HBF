#pragma once
#include <Device/cudaOverlayGraph.cuh>
#include <Graph.hpp>
#include <XLib.hpp>
#include <vector>
using namespace std;
using namespace graph;

namespace cuda_graph {

    void cudaGNRGraph::GetHostGraph()
    {
        vector<int>&Orders = gp.Orders;
        OverlayGraph& gt1 = upGraph;
        OverlayGraph& gt2 = downGraph;
        int upBias = 0,downBias = 0;
        for(int i=0;i<V;i++)
        {
            gt1.OutNodesVec[i].x = upBias;
            gt2.OutNodesVec[i].x = downBias;
            int s = gp.OutNodesVec[i];
            int t = gp.OutNodesVec[i+1];
            vector<int2> temp;
            int upT = 0,downT = 0;
            for(int j=s;j<t;j++)
            {
                int2 edge = gp.OutEdgesVec[j];
                if(Orders[edge.x]>Orders[i])
                {
                    gt1.OutEdgesVec.push_back(edge);
                    upT++;
                }
                else if(Orders[edge.x]<Orders[i])
                {
                    gt2.OutEdgesVec.push_back(edge);
                    downT++;
                }
                else if(Orders[edge.x]<(ALL_LEVELS-1))
                {
                    gt1.OutEdgesVec.push_back(edge);
                    upT++;
                    temp.push_back(edge);
                }
                else if(Orders[edge.x]==(ALL_LEVELS-1))
                {
                    gt1.OutEdgesVec.push_back(edge);
                    upT++;
                }
            }
            upBias += upT;
            downBias += downT;
            gt1.OutNodesVec[i].y = upBias;
#if NODESPLIT
            gt1.OutNodesVec[i].z = (upT / EdgesOneSplit + 1)>MAXSplits?MAXSplits:(upT / EdgesOneSplit+1);
#else
            gt1.OutNodesVec[i].z = 1;
#endif
            gt1.OutNodesVec[i].w = downT;
            gt2.OutNodesVec[i].w = downBias;
            gt2.OutNodesVec[i].z = 1;
            downBias +=  temp.size();
            gt2.OutNodesVec[i].y = downBias;
            for(int j=0;j<temp.size();j++)
                gt2.OutEdgesVec.push_back(temp[j]);
        }
        hostPrintfSmall(upBias,"upl");
        hostPrintfSmall(downBias,"downl");        
    }

    cudaGNRGraph::cudaGNRGraph(GraphPreDeal &_gp):gp(_gp)
    {
        V=gp.Orders.size();
        printf("newSize:%d\n",gp.OutEdgesVec.size());
#if LOGMARKS
        LOGIN<<"Orders: ";
        for(int i=0;i<gp.Orders.size();i++)
        {
            LOGIN<<gp.Orders[i]<<" ";
        }
        LOGIN<<std::endl;
        LOGIN<<"Marks: ";
        for(int i=0;i<gp.Marks.size();i++)
        {
            LOGIN<<gp.Marks[i]<<" ";
        }
        LOGIN<<std::endl;
#endif
        upGraph.resize(V);
        downGraph.resize(V);
        GetHostGraph();
        cudaStreamCreate(&upStream);
        cudaStreamCreate(&downStream);
#if LOGFRONTIER
        LOGIN<<"upGraphDegree: ";
        for(int i=0;i<upGraph.OutNodesVec.size()-1;i++)
        {
            LOGIN<<upGraph.OutNodesVec[i].y-upGraph.OutNodesVec[i].x<<" ";
        }
        LOGIN<<std::endl;
        LOGIN<<"downGraphDegree: ";
        for(int i=0;i<downGraph.OutNodesVec.size()-1;i++)
        {
            LOGIN<<downGraph.OutNodesVec[i].y-downGraph.OutNodesVec[i].x<<" ";
        }
        LOGIN<<std::endl;
#endif
    }

    void cudaGNRGraph::cudaMallocMem()
    {
        cudaMalloc(&upF1,16* V * sizeof(int));
        cudaMalloc(&upF1Size, 1 * sizeof(int));
        cudaMalloc(&upF2,16* V * sizeof(int));
        cudaMalloc(&upF2Size, 1 * sizeof(int));

        cudaMalloc(&downF1,16* V * sizeof(int));
        cudaMalloc(&downF1Size, 1 * sizeof(int));
        cudaMalloc(&downF2,16* V * sizeof(int));
        cudaMalloc(&downF2Size, 1 * sizeof(int));

        cudaMalloc(&devUpOutNodes, V * sizeof(int4));
        cudaMalloc(&devDownOutNodes, V * sizeof(int4));
        cudaMalloc(&devUpOutEdges, upGraph.OutEdgesVec.size() * sizeof(int2));
        cudaMalloc(&devDownOutEdges, downGraph.OutEdgesVec.size() * sizeof(int2));

        cudaMalloc(&devDistances, V * sizeof(hdist_t));
    }
    void cudaGNRGraph::cudaFreeMem()
    {
        cudaFree(upF1);
        cudaFree(upF1Size);
        cudaFree(upF2);
        cudaFree(upF2Size);
        
        cudaFree(downF1);
        cudaFree(downF1Size);
        cudaFree(downF2);
        cudaFree(downF2Size);

        cudaFree(devUpOutNodes);
        cudaFree(devDownOutNodes);
        cudaFree(devUpOutEdges);
        cudaFree(devDownOutEdges);

        cudaFree(devDistances);
        cudaStreamDestroy(upStream);
        cudaStreamDestroy(downStream);
    }
    void cudaGNRGraph::cudaCopyMem()
    {
        __CUDA_ERROR("copy");
        copyVecTo<int4>(upGraph.OutNodesVec,devUpOutNodes);
        copyVecTo<int4>(downGraph.OutNodesVec,devDownOutNodes);
        __CUDA_ERROR("copy");
        copyVecTo<int2>(upGraph.OutEdgesVec,devUpOutEdges);
        copyVecTo<int2>(downGraph.OutEdgesVec,devDownOutEdges);
        __CUDA_ERROR("copy");
    }
    
    int cudaGNRGraph::cudaInit(int initNode)
    {  
        vector<hdist_t> temp(V,INF);
        temp[initNode] = ZERO;
        copyVecTo<int2>(temp,devDistances);
        int4 tempNode = upGraph.OutNodesVec[initNode];
        vector<int> Nodes;
        for(int i=0;i<tempNode.z;i++)
        {
            int g = (unsigned)initNode | (i<<SplitBits);
            Nodes.push_back(g);
        }
        hostPrintfSmall(tempNode);
        int tn = Nodes.size();
        copyVecTo<int>(Nodes,upF1);
        copyIntTo(tn,upF1Size);
        Nodes.clear();
#if UPDOWNREM
        Nodes.push_back(initNode | 0x80000000);
#else
        Nodes.push_back(initNode);
#endif
        int td = Nodes.size();
        copyVecTo<int>(Nodes,downF1);
        copyIntTo(td,downF1Size);
        return tn;
    }
    void cudaGNRGraph::cudaClear()
    {
        int tn=0;
        copyIntTo(tn,upF1Size);
        copyIntTo(tn,upF2Size);
        copyIntTo(tn,downF1Size);
        copyIntTo(tn,downF2Size);
    }
    void cudaGNRGraph::cudaGetRes(vector<int>&res)
    {
        vector<int2> resT;
        resT.resize(res.size());
        copyVecFrom(resT,devDistances);
        for(int i=0;i<res.size();i++)
            res[i] = resT[i].y;
    }
}

