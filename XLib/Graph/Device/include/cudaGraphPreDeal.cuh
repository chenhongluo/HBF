/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

H-BF is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#pragma once

#include <XLib.hpp>
#include "../../Host/GraphHost.hpp"
using namespace graph;

namespace cuda_graph {

class cudaGraphPreDeal {
public:
    GraphPreDeal& graph;

    int *devOutNodes, *devInNodes;
    int2 *devInEdges,*devOutEdges;

    //for selecting nodes
    int* devCandidates;
    int* devCandidateSize;
    int* devOutLoads;
    int* devOutDones;
    int* devInLoads;
    int* devInDones;
    float* devMarks;

    int* devOrders;

    int3* devTriEdges;
    int* devTriEdgeSize;

    int3* devNewEdges;
    int* devNewEdgeSize;
    int3* devNewEndEdges;
    int* devNewEndEdgeSize;

private:
    // template<typename T>
    // void copyVecTo(T* dest,vector<T>& source)
    // {
    //     cudaMemcpy(dest, 
    //     &(source[0]), 
    //     (source.size()) * sizeof(T),
    //     cudaMemcpyHostToDevice);
    // }

    // template<typename T>
    // void copyVecFrom(T* source,vector<T>& dest)
    // {
    //     cudaMemcpy(&(dest[0]), 
    //     source, 
    //     (dest.size()) * sizeof(T),
    //     cudaMemcpyDeviceToHost);
    // }

    // void copyIntTo(int *dest,int source)
    // {
    //     cudaMemcpy(dest, 
    //     &source, 
    //     1 * sizeof(int),
    //     cudaMemcpyHostToDevice);
    // }

    // void copyIntFrom(int* source,int* dest)
    // {
    //     cudaMemcpy(dest, 
    //     source, 
    //     1 * sizeof(int),
    //     cudaMemcpyDeviceToHost);
    // }

    void copyNE(int* devNodes,int2* devEdges,
    vector<int>Nodes,vector<int2>Edges)
    {
        copyVecTo<int>(devNodes,Nodes);
        copyVecTo<int2>(devEdges,Edges);
    }

public:
    cudaGraphPreDeal(GraphPreDeal& _graph);

    ~cudaGraphPreDeal();
    void copyOut()
    {
        copyNE(devOutNodes,devOutEdges,graph.OutNodesVec,graph.OutEdgesVec);
    }
    void copyIn()
    {
        copyNE(devInNodes,devInEdges,graph.InNodesVec,graph.InEdgesVec);
    }
    void copyCandidate()
    {
        copyVecTo<int>(devCandidates,graph.Candidates);
        copyIntTo(devCandidateSize,graph.Candidates.size());
    }
    void copyOrders()
    {
        copyVecTo<int>(devOrders,graph.Orders);
    }
    void copyMarks()
    {
        copyVecFrom<float>(devMarks,graph.Marks);
    }
    void copyTriEdges()
    {
        copyVecTo<int3>(devTriEdges,graph.ContractedEdges);
        copyIntTo(devTriEdgeSize,graph.ContractedEdges.size());
    }
    void copyAddEdges()
    {
        int temp;
        copyIntFrom(devNewEndEdgeSize,&temp);
        graph.AddEdges.resize(temp);
        copyVecFrom<int3>(devNewEndEdges,graph.AddEdges);
    }
    void printfEdges(int3 *devEdges,int* len)
    {
        printf("print edges:\n");
        int temp;
        copyIntFrom(len,&temp);
        vector<int3> vt(temp);
        copyVecFrom<int3>(devEdges,vt);
        for(int i=0;i<vt.size();i++)
        {
            printf("x:%d,y:%d,z:%d\n",vt[i].x,vt[i].y,vt[i].z);
        }
    }

    void InitSize(int*psize)
    {
        int temp=0;
        copyIntTo(psize,temp);
    }

    void InitSize(int*psize,int &size)
    {
        copyIntTo(psize,size);
    }

    void copyLeftEdgesSize(int* psize,int &size)
    {
        copyIntFrom(psize,&size);
    }
};

} //@cuda_graph
