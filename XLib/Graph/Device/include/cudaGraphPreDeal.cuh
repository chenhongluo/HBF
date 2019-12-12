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

    // int3* devTriEdges;
    // int* devTriEdgeSize;

    // int3* devNewEdges;
    // int* devNewEdgeSize;
    // int3* devNewEndEdges;
    // int* devNewEndEdgeSize;

private:

    void copyNE(int* devNodes,int2* devEdges,vector<int>& Nodes,vector<int2>& Edges)
    {
        copyVecTo<int>(Nodes,devNodes);
        copyVecTo<int2>(Edges,devEdges);
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
        copyVecTo<int>(graph.Candidates,devCandidates);
        int temp = graph.Candidates.size();
        copyIntTo(temp,devCandidateSize);
    }
    void copyOrders()
    {
        copyVecTo<int>(graph.Orders,devOrders);
    }
    void copyMarks()
    {
        copyVecFrom<float>(graph.Marks,devMarks);
    }
};

} //@cuda_graph
