#include "../include/GraphPreDeal.hpp"
#include <algorithm>
#include <exception>

#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <vector>

using std::vector;

namespace graph
{
    static
    void ToCSR(vector<int3>& TriEdgesVec,
    vector<int>& Nodes,
    vector<int2>& Edges,
    int k)
    {
        int V = Nodes.size() - 1;
        vector<int> Degree(V,0);
        for(int i=0;i<TriEdgesVec.size();i++)
        {
            int index;
            if(k == 0)
                index = TriEdgesVec[i].x;
            else
                index = TriEdgesVec[i].y; 
            Degree[index]++;        
        }
        Nodes[0] = 0;
        std::partial_sum(Degree.begin(),Degree.end(),Nodes.begin()+1);
        vector<int> temp(V,0);
	    for (int i = 0; i < TriEdgesVec.size(); i++) 
        {
            int index,index_d;
            if(k == 0)
                index = TriEdgesVec[i].x;
            else
                index = TriEdgesVec[i].y; 
            const int bias = Nodes[index] + temp[index]++;
            if(k == 0)
                index_d = TriEdgesVec[i].y;
            else
                index_d = TriEdgesVec[i].x; 
            Edges[bias].x = index_d;
            Edges[bias].y = TriEdgesVec[i].z;
	    }
    }

    static void CSRMerge(vector<int> &Nodes1,
	vector<int2>& Edges1,
	vector<int>& Nodes2,
	vector<int2>& Edges2
    )
    {
        vector<int> Nodes(Nodes1.size());
        vector<int2> Edges(Edges1.size() + Edges2.size());
        for (int i = 0; i < Nodes1.size(); i++)
        {
            Nodes[i] = Nodes1[i] + Nodes2[i];
        }
        for (int i = 0; i < Nodes1.size()-1; i++)
        {
            int bias = Nodes1[i + 1] - Nodes1[i];
            std::copy(Edges1.begin()+Nodes1[i], Edges1.begin() + Nodes1[i + 1], Edges.begin() + Nodes[i]);
            std::copy(Edges2.begin() + Nodes2[i], Edges2.begin() + Nodes2[i + 1], Edges.begin() + Nodes[i] + bias);
        }
        Nodes2 = Nodes;
        Edges2 = Edges;
    }

    void CSRMerge(vector<int3>& TriEdgesVec,vector<int>& Nodes,
    vector<int2>& Edges,int k)
    {
        vector<int> tempV(Nodes.size(),0);
        vector<int2> tempE(TriEdgesVec.size());
        ToCSR(TriEdgesVec,tempV,tempE,k);
        CSRMerge(tempV,tempE,Nodes,Edges);
    }

} // namespace graph


