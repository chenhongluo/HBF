#pragma once

#include <../config.cuh>
#include <Host/GraphSSSP.hpp>
#include <Graph.hpp>
#include <vector>
using namespace graph;
using std::vector;

void PreDealGraph(GraphPreDeal& g);

namespace cuda_graph {
    class cudaBucket{
	public:
        int V;
        int* devF;
        int* devFSize;
    
    cudaBucket(const int _V);
    ~cudaBucket();
    void cudaMallocMem();
    void cudaFreeMem();
    void cudaCopyMem();
    void cudaClear();
    void printfBucket()
    {
        int temp;
        copyIntFrom(devFSize,&temp);
        vector<int> vt(temp);
        copyVecFrom<int>(devF,vt);
        // for(int i=0;i<vt.size();i++)
        // {
        //     printf("%d ",vt[i]);
        // }
        printf("%d\n",temp);
    }
    };

    class cudaGNRGraph{
    private:
        int V;
        GraphPreDeal &gp;
    public:
        vector<int> upOrderTrans;
        vector<int> downOrderTrans;
        int levels;

        int* devFt;
        int* devFtSize;
        int* devUpOrderTrans;
        int* devDownOrderTrans;

        int* devUpOutNodes;
        int* devDownOutNodes;
        int2* devUpOutEdges;
        int2* devDownOutEdges;

        int** devUpBuckets;
        int** devUpBucketSize;
        int** devDownBuckets;
        int** devDownBucketSize;

        hdist_t* devDistances;

        OverlayGraph upGraph;
        OverlayGraph downGraph;

        vector<cudaBucket> cudaUpBuckets;
        vector<cudaBucket> cudaDownBuckets;

        cudaGNRGraph(GraphPreDeal &_gp,const int _levels,
            const vector<int> upStrategy,const vector<int> downStrategy);

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
        void GetBuckets(const vector <int>&upStrategy,
        const vector<int>&downStrategy);

        int GetSize(const int *p)
        {
            int t;
            cudaMemcpy(&t, p, 1 * sizeof(int),cudaMemcpyDeviceToHost);
            return t;
        }

        void InitSize(int *p)
        {
            int tn = 0;
            cudaMemcpy(p, &tn, 1 * sizeof(int),cudaMemcpyHostToDevice);
        }

        void printfBuckets()
        {
            // printfStr("printfBuckets");
            // for(auto& x:cudaUpBuckets)
            // {
            //     x.printfBucket();
            // }
            for(auto& x:cudaDownBuckets)
            {
                x.printfBucket();
            }
        }

        void copyDistance(vector<int>& p)
        {
            vector<hdist_t> temp(V);
            copyVecFrom<hdist_t>(devDistances,temp);
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