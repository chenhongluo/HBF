#include <cub/cub.cuh>
#include <XLib.hpp>
#include <climits>
#include <vector>
#include <map>
#include <algorithm>
#include <../config.cuh>
#include <queue>
#include <cooperative_groups.h>

using namespace cooperative_groups;
using std::vector;
using std::map;
using std::queue;

namespace Kernels
{
    template<int VW_SIZE>
    __global__ void EdgeDifference(node_t* __restrict__ devNodes,
        int2* __restrict__ devEdges,
        int* __restrict__ devOrders,
        node_t* __restrict__ devCandidates,
        const int* pdevCandidateSize,
        int* devLoads,
        int* devDones)
    {
        thread_block g = this_thread_block();
		thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
        const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / tile.size();
        const int Stride = gridDim.x * (BLOCKDIM / tile.size());
        int DoneNum = 0,sum1 = 0;
        int LoadNum = 0,sum2 = 0;
        int devCandidateSize = cub::ThreadLoad<cub::LOAD_CG>(pdevCandidateSize);
        // if(blockIdx.x * BLOCKDIM + threadIdx.x == 0)
        // {
        //     printf("pdevCanSize:%d",devCandidateSize);
        // }
        int k=1;
        for (int i = VirtualID; i < devCandidateSize; i+=Stride)
        {
            tile.sync();
            const node_t index = cub::ThreadLoad<cub::LOAD_CG>(devCandidates+i);
            //const int myOrder = cub::ThreadLoad<cub::LOAD_LDG>(devOrders+index)
            const edge_t start = cub::ThreadLoad<cub::LOAD_CS>(devNodes+index);
            edge_t end = cub::ThreadLoad<cub::LOAD_CS>(devNodes+index+1);
            // if(k==1&&blockIdx.x * BLOCKDIM + threadIdx.x < 128)
            // {
            //     printf("index:%d,start:%d,end:%d,Id:%d\n",index,start,end,blockIdx.x * BLOCKDIM + threadIdx.x);
            //     k++;
            // }
            for (int k = start+tile.thread_rank();k<end+tile.thread_rank();k+=tile.size())
            {
                int2 dest;
                int destOrder;
                if(k<end)
                {
                    dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdges+k);
                    destOrder = cub::ThreadLoad<cub::LOAD_CG>(devOrders+dest.x);
                }
                tile.sync();
                if(k<end)
                {
                    //Error
                    if(destOrder<ALL_LEVELS-1)
                    {
                        ++DoneNum;
                        //++LoadNum;
                    }
                    else
                    {
                        ++LoadNum;
                        //atomicAdd(devInLoads+dest.x,1);
                    }
                }
            }
            VWInclusiveScanAdd<VW_SIZE,int>(tile,DoneNum,sum1);
            VWInclusiveScanAdd<VW_SIZE,int>(tile,LoadNum,sum2);
            if(tile.thread_rank() == tile.size()-1)
            {
                cub::ThreadStore<cub::STORE_CG>(devDones+index,sum1);
                cub::ThreadStore<cub::STORE_CG>(devLoads+index,sum2);
            }
            tile.sync();
            sum1 = DoneNum = sum2 = LoadNum = 0;
            //TODO chl
        }
        
    }

    const float alpha = 0.3;

    __global__ void NodeMarksCompute(
        int* __restrict__ devCandidates,
        const int* pdevCandidateSize,
        int* devOutLoads,
        int* devOutDones,
        int* devInLoads,
        int* devInDones,
        float* devMarks)
    {
        const int ID = blockIdx.x * BLOCKDIM + threadIdx.x;
        const int Stride = gridDim.x * BLOCKDIM;
        int devCandidateSize = cub::ThreadLoad<cub::LOAD_CG>(pdevCandidateSize);
        for (int i = ID; i < devCandidateSize; i+=Stride)
        {
            const int index = cub::ThreadLoad<cub::LOAD_CS>(devCandidates+i);
            //const int myOrder = cub::ThreadLoad<cub::LOAD_CS>(devOrders+index)
            const float inLoad = (float)cub::ThreadLoad<cub::LOAD_CS>(devInLoads+index);
            const float inDone = (float)cub::ThreadLoad<cub::LOAD_CS>(devInDones+index);
            const float outLoad = (float)cub::ThreadLoad<cub::LOAD_CS>(devOutLoads+index);
            const float outDone = (float)cub::ThreadLoad<cub::LOAD_CS>(devOutDones+index);
            float mark = (inLoad*outLoad-inLoad-outLoad) * (1-alpha) + alpha*(outDone+inDone);
            cub::ThreadStore<cub::STORE_CG>(devMarks+index,mark);
        }
        
    }
} // namespace Kernels__device__ __forceinline__ void EdgeDifference(const )


namespace PreDeal
{
    struct NodeMark
    {
        int node;
        float mark;
        bool operator<(const NodeMark& n) const;
    };


    
    bool NodeMark::operator<(const NodeMark& n) const
    {
        if (this->mark < n.mark)
        {
            return true;
        }
        return false;
    }
    using std::vector;
    //k==0:threshold,k=1:top
    // void SelectNodesThresholdOrTOP(
    //     vector<int>& Candidates,
    //     vector<int>& marks,
    //     vector<int>& orders,
    //     const int orderNow,
    //     const int thresholdOrTop,
    //     const int k);

    void SelectNodesThresholdOrTOP(
        vector<int>& Candidates,
        vector<float>& marks,
        vector<int>& orders,
        vector<int>& contractedNodes,
        const int orderNow,
        const int thresholdOrTop,
        const int k)
    {
        vector<NodeMark> nodeMarks(Candidates.size());
        for (int i = 0;i<Candidates.size();i++)
        {
            NodeMark temp = {Candidates[i],marks[Candidates[i]]};
            nodeMarks[i] = temp;
        }
        sort(nodeMarks.begin(),nodeMarks.end());
        contractedNodes.clear();
        int cnum = 0;
        if(k==0)
        {
            for (vector<NodeMark>::iterator i = nodeMarks.begin();i!=nodeMarks.end();i++)
            {
                if( (*i).mark < thresholdOrTop)
                {
                    orders[(*i).node] = orderNow;
                    contractedNodes.push_back((*i).node);
                    cnum++;
                }
            }
        }
        else if(k==1)
        {
            for (vector<NodeMark>::iterator i = nodeMarks.begin();i!=nodeMarks.begin()+thresholdOrTop;i++)
            {
                orders[(*i).node] = orderNow;
                contractedNodes.push_back((*i).node);
                cnum++;
            }
        }
        Candidates.resize(Candidates.size()-cnum);
        int b=0;
        for (int i=0;i<orders.size();i++)
        {
            if(orders[i] == ALL_LEVELS-1)
            {
                Candidates[b++] = i;
            }
        }
    }

    struct cmp
    {
        bool operator()(int2 &a, int2 &b) const
        {
            return a.y > b.y;
        }
    };

    static void printfQueue(std::priority_queue<int2,vector<int2>,cmp> q)
    {
        printf("queue:\n");
        while(!q.empty())
        {
            int2 t = q.top();
            q.pop();
            printInt2(t);
        }
    }

    static void printfMap(map<int,int> dis)
    {
        printf("Map:\n");
        for(map<int,int>::iterator iter = dis.begin(); iter != dis.end(); iter++)
        {
            printf("%d,%d\n",iter->first,iter->second);
        }
    }

    void ShortCutCPU(vector<int> &ContractedNodes,vector<int> &Orders,vector<int3>& newEdges,vector<int>& OutNodes,vector<int2> &OutEdges,
    vector<int>& InNodes,vector<int2>& InEdges)
    {
        for(int i=0;i<ContractedNodes.size();i++)
        {
            int node = ContractedNodes[i];
            int myOrder = Orders[node];
            std::priority_queue<int2,vector<int2>,cmp> qp;
            map<int,int> dis;
            dis.insert(map<int, int>::value_type(node,0));
            qp.push(make_int2(node,0));
            // printfQueue(qp);
            while(!qp.empty())
            {
                int2 nw = qp.top();
                qp.pop();
                // printfQueue(qp);
                // printfMap(dis);
                if(nw.y>(dis.find(nw.x))->second)
                {
                    //printf("continue\n");
                    continue;
                }
                for(int s=OutNodes[nw.x];s<OutNodes[nw.x+1];s++)
                {
                    int2 edge = OutEdges[s];
                    if(Orders[edge.x] == myOrder)
                    {
                        map<int,int>::iterator it=dis.find(edge.x);
                        if(it!=dis.end() && it->second>(nw.y+edge.y))
                        {
                            it->second = (nw.y+edge.y);
                            qp.push(make_int2(edge.x,nw.y+edge.y));
                        }
                        else if(it == dis.end())
                        {
                            dis.insert(map<int, int>::value_type(edge.x,nw.y+edge.y));
                            qp.push(make_int2(edge.x,nw.y+edge.y));
                        }
                        // printfQueue(qp);
                        // printfMap(dis);
                    }
                }
            }
            // printStr("newEdges");
            // printVector(newEdges,newEdges.size());
            // printInt(Orders[node]);
            for(map<int,int>::iterator iter = dis.begin(); iter != dis.end(); iter++)
            {

                int myOrder = Orders[node];
                for(int s=InNodes[node];s<InNodes[node+1];s++)
                {
                    int2 inedge = InEdges[s];
                    // printStr("inedge");
                    // printInt2(inedge);
                    // printInt(Orders[inedge.x]);
                    if(Orders[inedge.x]>myOrder)
                    {
                        for(int t=OutNodes[iter->first];t<OutNodes[iter->first+1];t++)
                        {
                            int2 outedge = OutEdges[t];
                            // printStr("outedge");
                            // printInt2(outedge);
                            // printInt(Orders[outedge.x]);
                            if(Orders[outedge.x]>myOrder && inedge.x != outedge.x)
                            {
                                newEdges.push_back(make_int3(inedge.x,outedge.x,inedge.y+iter->second+outedge.y));
                            }
                        }
                    }
                }
            }
            // printStr("newEdges");
            // printVector(newEdges,newEdges.size());
        }
    }

} // namespace predeal for cpu


namespace Kernels
{
    template<int VW_SIZE,typename T>
    __device__ __forceinline__
    void VWWrite(thread_block_tile<VW_SIZE>& tile,int *pAllsize,T* writeStartAddr,
            const int& writeCount,T* data)
    {
        int sum = 0;
        int bias = 0;
        VWInclusiveScanAdd<VW_SIZE,int>(tile,writeCount,sum);
        if(tile.thread_rank() == tile.size()-1 && sum !=0)
        {
            //printf("fdsasbias:%d,%d\n",sum,bias,tile.thread_rank());
            bias = atomicAdd(pAllsize,sum);
            //printf("dfsafdbias:%d,%d\n",sum,bias,tile.thread_rank());
        }
        bias = tile.shfl(bias,tile.size()-1);
        sum -= writeCount;
        for(int it = 0;it<writeCount;it++)
        {
            cub::ThreadStore<cub::STORE_CG>(writeStartAddr+bias+sum+it,data[it]);
            
        }
    }
    template<int VW_SIZE>
    __global__
    void ShortCut(
    const int3* devTriEdges,
    const int* pdevTriEdgeSize,
    node_t* __restrict__ devNodes,
    int2* __restrict__ devEdges,
    int* __restrict__ devOrders,
    int3* devNewEdges,
    int* devNewEdgeSize,
    int3* devNewEndEdges,
    int* devNewEndEdgeSize)
    {
        thread_block g = this_thread_block();
        thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
        const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / tile.size();
        const int Stride = gridDim.x * (BLOCKDIM / tile.size());
        const int devTriEdgeSize = cub::ThreadLoad<cub::LOAD_CS>(pdevTriEdgeSize);
        int tempEdgeNum = 0;
        int tempEndEdgeNum = 0;
        int3 tempEdge[8];
        int3 tempEndEdge[4];
        // if(blockIdx.x * BLOCKDIM + threadIdx.x == 0)
        // {
        //     printf("devTriEdgeSize:%d\n",devTriEdgeSize);
        // }
        int ttt=0;
        for (int i = VirtualID; i < devTriEdgeSize; i+=Stride)
        {
            int3 edge = cub::ThreadLoad<cub::LOAD_CS> (devTriEdges+i);
            const edge_t start = cub::ThreadLoad<cub::LOAD_CS>(devNodes+edge.y);
            edge_t end = cub::ThreadLoad<cub::LOAD_CS>(devNodes+edge.y+1);
            int myOrder = cub::ThreadLoad<cub::LOAD_CG>(devOrders+edge.y);
            for (int k = start+tile.thread_rank();k<end+tile.thread_rank();k+=tile.size())
            {
                int2 dest;
                int destOrder;
                if(k<end)
                {
                    dest = cub::ThreadLoad<cub::LOAD_CS>(devEdges+k);
                    destOrder = cub::ThreadLoad<cub::LOAD_CG>(devOrders+dest.x);
                    // if(destOrder == 0)
                    // {
                    //     printf("dfsagfhasjfhajfdhshdh\n");
                    // }
                    // printf("k:%d\n",k);
                    // printf("%d,%d,%d\n",dest.x,dest.y,destOrder);
                }
                tile.sync();
                // if(ttt == 0 && blockIdx.x * BLOCKDIM + threadIdx.x <256 && tile.thread_rank()==0)
                // {
                //     ttt++;
                //     int2 ggg1 = cub::ThreadLoad<cub::LOAD_CS>(devEdges);
                //     int2 ggg2 = cub::ThreadLoad<cub::LOAD_CS>(devEdges+1396468);
                //     printf("%d,%d\n",ggg1.x,ggg2.x);
                //     printf("%d,%d,%d,%d,%d,%d,%d\n",start,end,edge.x,edge.y,edge.z,dest.x,dest.y);
                //     printf("destOrder:%d,myOrder:%d\n",destOrder,myOrder);
                // }
                if(k<end)
                {
                    if(destOrder > myOrder  && edge.x!=dest.x )
                    {
                        tempEndEdge[tempEndEdgeNum++] = make_int3(edge.x,dest.x,edge.z+dest.y);

                    }
                    else if(destOrder == myOrder)
                    {
                        //printf("dfsagfhasjfhajfdhshdh\n");
                        tempEdge[tempEdgeNum++] = make_int3(edge.x,dest.x,edge.z+dest.y);
                    }
                }
                if(tile.any(tempEdgeNum>=8))
                {
                    VWWrite<VW_SIZE,int3>(tile,devNewEdgeSize,devNewEdges,tempEdgeNum,tempEdge);
                    tempEdgeNum = 0;
                }
                if(tile.any(tempEndEdgeNum>=4))
                {
                    VWWrite<VW_SIZE,int3>(tile,devNewEndEdgeSize,devNewEndEdges,tempEndEdgeNum,tempEndEdge);
                    tempEndEdgeNum = 0;
                }
                tile.sync();
            }
        }
        // for(int i=0;i<tempEdgeNum;i++)
        // {
        //     printf("tempEdge:%d,%d,%d",tempEdge[i].x,tempEdge[i].y,tempEdge[i].z);
        // }
        VWWrite<VW_SIZE,int3>(tile,devNewEdgeSize,devNewEdges,tempEdgeNum,tempEdge);
        VWWrite<VW_SIZE,int3>(tile,devNewEndEdgeSize,devNewEndEdges,tempEndEdgeNum,tempEndEdge);
        tempEdgeNum = tempEndEdgeNum = 0;
    }
}
namespace PreDeal
{
    void GetTriEdges(const vector<int>& ContractedNodes,
    const vector<int>&Orders,
    const int myOrder,
    const vector<int>&InNodes,
    const vector<int2>&InEdges,
    vector<int3>&TriEdges
    )
    {
        TriEdges.clear();
        int cnum=0;
        for(int i=0;i<ContractedNodes.size();i++)
        {
            int s=InNodes[ContractedNodes[i]];
            int t=InNodes[ContractedNodes[i]+1];
            for(int k=s;k<t;k++)
            {
                //int3 temp = make_int3(InEdges[k].x,ContractedNodes[i],InEdges[k].y);
                //printf("%d,%d,%d\n",InEdges[k].x,ContractedNodes[i],InEdges[k].y);
                if(Orders[InEdges[k].x] > myOrder)
                    cnum++;
            }
            //printf("%d,%d,%d\n",ContractedNodes[i],s,t);

        }
        //printf("%d\n",cnum);
        TriEdges.resize(cnum);
        int b=0;
        for(int i=0;i<ContractedNodes.size();i++)
        {
            int s=InNodes[ContractedNodes[i]];
            int t=InNodes[ContractedNodes[i]+1];
            for(int k=s;k<t;k++)
            {
                if(Orders[InEdges[k].x] > myOrder)
                {
                    int3 temp = make_int3(InEdges[k].x,ContractedNodes[i],InEdges[k].y);
                    //printf("%d,%d,%d\n",InEdges[k].x,ContractedNodes[i],InEdges[k].y);
                    TriEdges[b++] = temp;
                }
            }
        }
    }
}