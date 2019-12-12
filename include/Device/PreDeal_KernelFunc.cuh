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
            // float mark = (inLoad*outLoad-inLoad-outLoad) * (1-alpha) + alpha*(outDone+inDone);
            float mark = (inLoad*outLoad ) * (1-alpha) + alpha*(outDone*inDone);
            if(outLoad < 8)
                mark = outLoad * belta + mark; 
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

    // static void printfQueue(std::priority_queue<int2,vector<int2>,cmp> q)
    // {
    //     printf("queue:\n");
    //     while(!q.empty())
    //     {
    //         int2 t = q.top();
    //         q.pop();
    //         printInt2(t);
    //     }
    // }

    // static void printfMap(map<int,int> dis)
    // {
    //     printf("Map:\n");
    //     for(map<int,int>::iterator iter = dis.begin(); iter != dis.end(); iter++)
    //     {
    //         printf("%d,%d\n",iter->first,iter->second);
    //     }
    // }

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
            while(!qp.empty())
            {
                int2 nw = qp.top();
                qp.pop();
                if(nw.y>(dis.find(nw.x))->second)
                {
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
                    }
                }
            }
            for(map<int,int>::iterator iter = dis.begin(); iter != dis.end(); iter++)
            {

                int myOrder = Orders[node];
                for(int s=InNodes[node];s<InNodes[node+1];s++)
                {
                    int2 inedge = InEdges[s];
                    if(Orders[inedge.x]>myOrder)
                    {
                        for(int t=OutNodes[iter->first];t<OutNodes[iter->first+1];t++)
                        {
                            int2 outedge = OutEdges[t];
                            if(Orders[outedge.x]>myOrder && inedge.x != outedge.x)
                            {
                                newEdges.push_back(make_int3(inedge.x,outedge.x,inedge.y+iter->second+outedge.y));
                            }
                        }
                    }
                }
            }
        }
    }

} // namespace predeal for cpu