#pragma once
#include <Device/cudaOverlayGraph.cuh>
using namespace std;
using namespace graph;

__constant__ int2 *buckets[MAXDistanceSplit];
__constant__ int *bucketSizes[MAXDistanceSplit];
namespace cuda_graph
{
void cudaGNRGraph::GetHostGraph()
{
    vector<int> &Orders = gp.Orders;
    OverlayGraph &gt1 = upGraph;
    OverlayGraph &gt2 = downGraph;
    int upBias = 0, downBias = 0;
    for (int i = 0; i < V; i++)
    {
        gt1.OutNodesVec[i].x = upBias;
        gt2.OutNodesVec[i].x = downBias;
        int s = gp.OutNodesVec[i];
        int t = gp.OutNodesVec[i + 1];
        vector<int2> temp;
        int upT = 0, downT = 0;
        for (int j = s; j < t; j++)
        {
            int2 edge = gp.OutEdgesVec[j];
            devDistanceMaxBucketSizes[edge.x / DistanceSplitSize]++;
            if (Orders[edge.x] > Orders[i])
            {
                gt1.OutEdgesVec.push_back(edge);
                upT++;
            }
            else if (Orders[edge.x] < Orders[i])
            {
                gt2.OutEdgesVec.push_back(edge);
                downT++;
            }
            else if (Orders[edge.x] < (ALL_LEVELS - 1))
            {
                gt1.OutEdgesVec.push_back(edge);
                upT++;
                temp.push_back(edge);
            }
            else if (Orders[edge.x] == (ALL_LEVELS - 1))
            {
                gt1.OutEdgesVec.push_back(edge);
                upT++;
            }
        }
        upBias += upT;
        downBias += downT;
        gt1.OutNodesVec[i].y = upBias;
        gt1.OutNodesVec[i].z = 1;
        gt1.OutNodesVec[i].w = downT;
        gt2.OutNodesVec[i].w = downBias;
        gt2.OutNodesVec[i].z = 1;
        downBias += temp.size();
        gt2.OutNodesVec[i].y = downBias;
        for (int j = 0; j < temp.size(); j++)
            gt2.OutEdgesVec.push_back(temp[j]);
    }
    printf("bucket:%d\n", devDistanceMaxBucketSizes.size());
    for (int x : devDistanceMaxBucketSizes)
        printf("%d ", x);
    printf("\n");
    hostPrintfSmall(upBias, "upEdges:");
    hostPrintfSmall(downBias, "downDownEdges");
}

cudaGNRGraph::cudaGNRGraph(GraphPreDeal &_gp) : gp(_gp)
{
    V = gp.Orders.size();
    int maxcb = (V % DistanceSplitSize == 0) ? (V / DistanceSplitSize) : (V / DistanceSplitSize + 1);
    devDistanceMaxBucketSizes.resize(maxcb);
    for (int &x : devDistanceMaxBucketSizes)
        x = 0;
    printf("newSize:%d\n", gp.OutEdgesVec.size());
#if LOGMARKS
    LOGIN << "Orders: ";
    for (int i = 0; i < gp.Orders.size(); i++)
    {
        LOGIN << gp.Orders[i] << " ";
    }
    LOGIN << std::endl;
    LOGIN << "Marks: ";
    for (int i = 0; i < gp.Marks.size(); i++)
    {
        LOGIN << gp.Marks[i] << " ";
    }
    LOGIN << std::endl;
#endif
    upGraph.resize(V);
    downGraph.resize(V);
    GetHostGraph();
    cudaStreamCreate(&upStream);
    cudaStreamCreate(&downStream);
#if LOGFRONTIER
    LOGIN << "upGraphDegree: ";
    for (int i = 0; i < upGraph.OutNodesVec.size() - 1; i++)
    {
        LOGIN << upGraph.OutNodesVec[i].y - upGraph.OutNodesVec[i].x << " ";
    }
    LOGIN << std::endl;
    LOGIN << "downGraphDegree: ";
    for (int i = 0; i < downGraph.OutNodesVec.size() - 1; i++)
    {
        LOGIN << downGraph.OutNodesVec[i].y - downGraph.OutNodesVec[i].x << " ";
    }
    LOGIN << std::endl;
#endif
}

void cudaGNRGraph::cudaMallocMem()
{
    cudaMalloc(&upF1, 1 * V * sizeof(int));
    cudaMalloc(&upF2, 1 * V * sizeof(int));
    cudaMalloc(&downF1, 1 * V * sizeof(int));
    cudaMalloc(&downF2, 1 * V * sizeof(int));

    cudaMalloc(&devSizeTemp, 4 * sizeof(int));
    upF1Size = devSizeTemp;
    upF2Size = devSizeTemp + 1;
    downF1Size = devSizeTemp + 2;
    downF2Size = devSizeTemp + 3;
    // cudaMalloc(&upF2Size, 1 * sizeof(int));
    // cudaMalloc(&upF1Size, 1 * sizeof(int));
    // cudaMalloc(&downF1Size, 1 * sizeof(int));
    // cudaMalloc(&downF2Size, 1 * sizeof(int));

    int *bucketSizesTemp[MAXDistanceSplit];
#if DISTANCESPLIT
    cudaMalloc(&devDistanceBucketSizes, MAXDistanceSplit * sizeof(int));
    for (int i = 0; i < MAXDistanceSplit; i++)
        bucketSizesTemp[i] = devDistanceBucketSizes + i;
    cudaMemcpyToSymbol(bucketSizes, bucketSizesTemp, sizeof(int *) * MAXDistanceSplit);
    for (int i = 0; i < devDistanceMaxBucketSizes.size(); i++)
        cudaMalloc(&(devDistanceBuckets[i]), devDistanceMaxBucketSizes[i] * sizeof(int2));
    cudaMemcpyToSymbol(buckets, devDistanceBuckets, sizeof(int2 *) * devDistanceMaxBucketSizes.size());
#endif

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

    cudaFree(devSizeTemp);
    // cudaFree(downF1);
    // cudaFree(downF1Size);
    // cudaFree(downF2);
    // cudaFree(downF2Size);

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
    copyVecTo<int4>(upGraph.OutNodesVec, devUpOutNodes);
    copyVecTo<int4>(downGraph.OutNodesVec, devDownOutNodes);
    __CUDA_ERROR("copy");
    copyVecTo<int2>(upGraph.OutEdgesVec, devUpOutEdges);
    copyVecTo<int2>(downGraph.OutEdgesVec, devDownOutEdges);
    __CUDA_ERROR("copy");
}

int cudaGNRGraph::cudaInit(int initNode)
{
    //init devDistance
    vector<hdist_t> temp(V, INF);
    temp[initNode] = ZERO;
    copyVecTo<int2>(temp, devDistances);

    vector<int> Nodes;
    Nodes.push_back(initNode);
    int tn = Nodes.size();
    copyVecTo<int>(Nodes, upF1);
    copyIntTo(tn, upF1Size);
    return tn;
}
void cudaGNRGraph::cudaClearBucket()
{
#if DISTANCESPLIT
    vector<int> tn(MAXDistanceSplit, 0);
    copyVecTo<int>(tn, devDistanceBucketSizes);
#endif
}

void cudaGNRGraph::cudaClear()
{
    vector<int> tn(4, 0);
    copyVecTo<int>(tn, devSizeTemp);
}
void cudaGNRGraph::cudaGetRes(vector<int> &res)
{
    vector<int2> resT;
    resT.resize(res.size());
    copyVecFrom(resT, devDistances);
    for (int i = 0; i < res.size(); i++)
        res[i] = resT[i].y;
}

void cudaGNRGraph::BucketDebug()
{
#if DISTANCESPLIT
    vector<int> sizes;
    sizes.resize(MAXDistanceSplit);
    copyVecFrom(sizes, devDistanceBucketSizes);
    for (int i = 0; i < 15; i++)
    {
        printf("%d ", sizes[i]);
    }
    printf("\n");
    vector<int2> realdata;
    for (int i = 0; i < 8; i++)
    {
        realdata.resize(sizes[i]);
        copyVecFrom<int2>(realdata, devDistanceBuckets[i]);
        if (sizes[i] < 10)
        {
            for (int j = 0; j < sizes[i]; j++)
                printf("%d %d   ", realdata[j].x, realdata[j].y);
        }
    }
#endif
}
} // namespace cuda_graph
