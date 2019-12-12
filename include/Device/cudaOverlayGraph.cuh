#pragma once

#include <../config.cuh>
#include <Host/GraphSSSP.hpp>
#include <Graph.hpp>
#include <vector>
using namespace graph;
using std::vector;

void PreDealGraph(GraphPreDeal& g);

namespace cuda_graph {
    class cudaGNRGraph{
    private:
        int V;
        GraphPreDeal &gp;
    public:
        cudaStream_t upStream,downStream;

        int *upF1,*upF1Size,*upF2,*upF2Size;
        int *downF1,*downF1Size,*downF2,*downF2Size;

        int4* devUpOutNodes;
        int4* devDownOutNodes;
        int2* devUpOutEdges;
        int2* devDownOutEdges;

        hdist_t* devDistances;

        OverlayGraph upGraph;
        OverlayGraph downGraph;
        
        cudaGNRGraph(GraphPreDeal &_gp);
        
        void cudaMallocMem();
        void cudaFreeMem();
        void cudaCopyMem();
        int cudaInit(int initNode);
        void cudaClear();
        void cudaGetRes(vector<int>&res);
        void GNRSearchMain(int source);
        void WorkEfficient(GraphWeight& graph);
        void FrontierDebug(const int FSize, const int level);
    private:
        void GetHostGraph();
        void copyDistance(vector<int>& p)
        {
            vector<hdist_t> temp(V);
            copyVecFrom<hdist_t>(temp,devDistances);
#if ATOMIC64
            for(int i=0;i<temp.size();i++)
            {
                // printInt2(temp[i]);
                p[i] = temp[i].y;
            }
#else
            for(int i=0;i<temp.size();i++)
            {
                p[i] = temp[i];
            }
#endif
        }
    };
}