#pragma once
#include <Device/cudaOverlayGraph.cuh>
#include <Graph.hpp>
#include <XLib.hpp>
#include <vector>
using namespace std;
using namespace graph;

namespace cuda_graph {
    cudaBucket::cudaBucket(const int _V):
    V(_V)
    {
    }
    cudaBucket::~cudaBucket()
    {
    }

    void cudaBucket::cudaMallocMem()
    {
        cudaMalloc(&devF, V * sizeof(int));
        cudaMalloc(&devFSize, 1 * sizeof(int));
    }

    void cudaBucket::cudaFreeMem()
    {
        cudaFree(devF);
        cudaFree(devFSize);
        // cudaFree(devOutEdges);
        // cudaFree(devOutNodes);
    }

    void cudaBucket::cudaCopyMem()
    {
        // cudaMemcpy(devOutEdges,&g.OutEdgesVec[0],  g.OutEdgesVec.size() * sizeof(int3), 
        //  cudaMemcpyHostToDevice);
        // cudaMemcpy(devOutNodes,&g.OutNodesVec[0],  g.OutNodesVec.size() * sizeof(int), 
        //  cudaMemcpyHostToDevice);         
    }
    void cudaBucket::cudaClear()
    {
        int tn = 0;
        cudaMemcpy(devFSize,&tn,1*sizeof(int),cudaMemcpyHostToDevice); 
    }

    inline void   getK(const vector <int>&Strategy,
    vector<int> &orderTrans)
    {
        int i=0,k=0;
        for(;i<orderTrans.size();i++)
        {
            if (i<Strategy[k])
            {
                orderTrans[i] = k;
            }
            else
            {
                orderTrans[i] = k++;
            }
            
        }
    }

    void cudaGNRGraph::GetBuckets(const vector <int>&upStrategy,
        const vector<int>&downStrategy)
    {
        for(int x:upStrategy)
        {
            cudaBucket cb(V);
            cudaUpBuckets.push_back(cb);
        }
        for(int x:downStrategy)
        {
            cudaBucket cb(V);
            cudaDownBuckets.push_back(cb);
        }
        vector<int> upOrder2(levels);
        vector<int> downOrder2(levels);
        getK(upStrategy,upOrder2);
        getK(downStrategy,downOrder2);
        int upBias=0,downBias=0;

        vector<int>&Orders = gp.Orders;
        OverlayGraph& gt1 = upGraph;
        OverlayGraph& gt2 = downGraph;
        gt1.OutNodesVec[0] = 0;
        gt2.OutNodesVec[0] = 0;
        for(int i=0;i<V;i++)
        {
            upOrderTrans[i] = upOrder2[Orders[i]];
            downOrderTrans[i] = downOrder2[Orders[i]];
            int s = gp.OutNodesVec[i];
            int t = gp.OutNodesVec[i+1];
            for(int j=s;j<t;j++)
            {
                int2 edge = gp.OutEdgesVec[j];
                if(Orders[edge.x]>Orders[i])
                {
                    gt1.OutEdgesVec.push_back(edge);
                    upBias++;
                    //gt1.OutEdgesVec[upBias++]={edge.x,edge.y};
                }
                else if(Orders[edge.x]<Orders[i])
                {
                    gt2.OutEdgesVec.push_back(edge);
                    downBias++;
                    //gt2.OutEdgesVec[downBias++]={edge.x,edge.y};
                }
                else
                {
                    gt1.OutEdgesVec.push_back(edge);
                    upBias++;
                    gt2.OutEdgesVec.push_back(edge);
                    downBias++;
                }
                  
            }
            gt1.OutNodesVec[i+1] = upBias;
            gt2.OutNodesVec[i+1] = downBias;
        }
    }

    cudaGNRGraph::cudaGNRGraph(GraphPreDeal &_gp,const int _levels,
      const vector<int> upStrategy,const vector<int> downStrategy):gp(_gp),levels(_levels)
    {
        V=gp.Orders.size();
        printf("newSize:%d\n",gp.OutEdgesVec.size());
        upOrderTrans.resize(V);
        downOrderTrans.resize(V);
        upGraph.resize(V);
        downGraph.resize(V);
        GetBuckets(upStrategy,downStrategy);
    }

    void cudaGNRGraph::cudaMallocMem()
    {
        cudaMalloc(&devFt, V * sizeof(int));
        cudaMalloc(&devFtSize, 1 * sizeof(int));

        cudaMalloc(&devUpOrderTrans, V * sizeof(int));
        cudaMalloc(&devDownOrderTrans, V * sizeof(int));
        cudaMalloc(&devUpOutNodes, (V+1) * sizeof(int));
        cudaMalloc(&devDownOutNodes, (V+1) * sizeof(int));
        cudaMalloc(&devUpOutEdges, upGraph.OutEdgesVec.size() * sizeof(int2));
        cudaMalloc(&devDownOutEdges, downGraph.OutEdgesVec.size() * sizeof(int2));
        
        cudaMalloc(&devUpBuckets, cudaUpBuckets.size() * sizeof(int*));
        cudaMalloc(&devUpBucketSize, cudaUpBuckets.size() * sizeof(int*));
        cudaMalloc(&devDownBuckets, cudaDownBuckets.size() * sizeof(int*));
        cudaMalloc(&devDownBucketSize, cudaDownBuckets.size() * sizeof(int*));

        cudaMalloc(&devDistances, V * sizeof(hdist_t));
        
        for(auto& x:cudaUpBuckets)
        {
            x.cudaMallocMem();
        }
        for(auto& x:cudaDownBuckets)
        {
            x.cudaMallocMem();
        }
    }
    void cudaGNRGraph::cudaFreeMem()
    {
        cudaFree(devFt);
        cudaFree(devFtSize);
        cudaFree(devDistances);
        cudaFree(devUpOrderTrans);
        cudaFree(devDownOrderTrans);
        cudaFree(devUpOutNodes);
        cudaFree(devDownOutNodes);
        cudaFree(devUpOutEdges);
        cudaFree(devDownOutEdges);
        cudaFree(devUpBuckets);
        cudaFree(devUpBucketSize);
        cudaFree(devDownBuckets);
        cudaFree(devDownBucketSize);
        for(auto& x:cudaUpBuckets)
        {
            x.cudaFreeMem();
        }
        for(auto& x:cudaDownBuckets)
        {
            x.cudaFreeMem();
        }
    }
    void cudaGNRGraph::cudaCopyMem()
    {
        cudaMemcpy(devUpOrderTrans, &upOrderTrans[0], V * sizeof (int), cudaMemcpyHostToDevice);
        cudaMemcpy(devDownOrderTrans, &upOrderTrans[0], V * sizeof (int), cudaMemcpyHostToDevice);
        __CUDA_ERROR("copy");
        cudaMemcpy(devUpOutNodes, &upGraph.OutNodesVec[0], (V+1) * sizeof (int), cudaMemcpyHostToDevice);
        cudaMemcpy(devDownOutNodes, &downGraph.OutNodesVec[0], (V+1) * sizeof (int), cudaMemcpyHostToDevice);
        __CUDA_ERROR("copy");
        cudaMemcpy(devUpOutEdges, &upGraph.OutEdgesVec[0], upGraph.OutEdgesVec.size() * sizeof (int2), cudaMemcpyHostToDevice);
        cudaMemcpy(devDownOutEdges, &downGraph.OutEdgesVec[0], downGraph.OutEdgesVec.size() * sizeof (int2), cudaMemcpyHostToDevice);
        __CUDA_ERROR("copy");
        int tn=cudaUpBuckets.size();
        vector<int*> Bt(tn);
        vector<int*> St(tn);

        for(int i=0;i<tn;i++)
        {
            Bt[i] = cudaUpBuckets[i].devF;
            St[i] = cudaUpBuckets[i].devFSize;
        }
        cudaMemcpy(devUpBuckets, &Bt[0],tn * sizeof (int*), cudaMemcpyHostToDevice);
        cudaMemcpy(devUpBucketSize, &St[0],tn  * sizeof (int*), cudaMemcpyHostToDevice);
        __CUDA_ERROR("copy");

        int dn=cudaDownBuckets.size();
        vector<int*> dBt(dn);
        vector<int*> dSt(dn);
        for(int i=0;i<dn;i++)
        {
            dBt[i] = cudaDownBuckets[i].devF;
            dSt[i] = cudaDownBuckets[i].devFSize;
        }
        cudaMemcpy(devDownBuckets, &dBt[0],dn * sizeof (int*), cudaMemcpyHostToDevice);
        cudaMemcpy(devDownBucketSize, &dSt[0],dn * sizeof (int*), cudaMemcpyHostToDevice);
        __CUDA_ERROR("copy");
        
        // for(auto& x:cudaDownBuckets)
        // {
        //     x.cudaCopyMem();
        // }
        // for(auto& x:cudaUpBuckets)
        // {
        //     x.cudaCopyMem();
        // }
    }
    
    int cudaGNRGraph::cudaInit(int initNode)
    {  
        vector<hdist_t> temp(V,INF);
        cudaMemcpy(devDistances, &temp[0], V * sizeof (hdist_t), cudaMemcpyHostToDevice);
	    const hdist_t zero = ZERO;
		cudaMemcpy(devDistances + initNode, &zero, sizeof(hdist_t), cudaMemcpyHostToDevice);
        for(auto& x:cudaUpBuckets)
        {
            x.cudaClear();
        }
        for(auto& x:cudaDownBuckets)
        {
            x.cudaClear();
        }
        int initBucket = upOrderTrans[initNode];
        int tn = 1;
        cudaMemcpy(cudaUpBuckets[initBucket].devF, &initNode, 1 * sizeof (int), cudaMemcpyHostToDevice);
	    cudaMemcpy(cudaUpBuckets[initBucket].devFSize, &tn, 1 * sizeof (int), cudaMemcpyHostToDevice);
        int down_ = downOrderTrans[initNode];
        tn = 1;
        cudaMemcpy(cudaDownBuckets[down_].devF, &initNode, 1 * sizeof (int), cudaMemcpyHostToDevice);
	    cudaMemcpy(cudaDownBuckets[down_].devFSize, &tn, 1 * sizeof (int), cudaMemcpyHostToDevice);
        
        return initBucket;
    }
    void cudaGNRGraph::cudaClear()
    {
        int tn = 0;
        cudaMemcpy(devFtSize,&tn,1*sizeof(int),cudaMemcpyHostToDevice);
        for(auto& x:cudaUpBuckets)
        {
            x.cudaClear();
        }
        for(auto& x:cudaDownBuckets)
        {
            x.cudaClear();
        }
    }
    void cudaGNRGraph::cudaGetRes(vector<int>&res)
    {
        cudaMemcpy(&(res[0]), devDistances, (res.size()) * sizeof(int),cudaMemcpyDeviceToHost);
    }
}

