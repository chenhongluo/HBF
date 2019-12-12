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

#include "../include/cudaGraphPreDeal.cuh"

namespace cuda_graph {

cudaGraphPreDeal::cudaGraphPreDeal(GraphPreDeal& _graph) :
                                 graph(_graph){

	cudaMalloc(&devOutNodes, (graph.V + 1) * sizeof(int));
	cudaMalloc(&devOutEdges, graph.E * sizeof(int2)*5);

	cudaMalloc(&devOutLoads, graph.V * sizeof(int));
    cudaMalloc(&devOutDones, graph.V * sizeof(int));
	cudaMalloc(&devMarks, graph.V * sizeof(int));
        
	cudaMalloc(&devOrders, graph.V * sizeof(int));
	cudaMalloc(&devCandidates, graph.V * sizeof(int));
	cudaMalloc(&devCandidateSize, 1 * sizeof(int));

	// cudaMalloc(&devTriEdges, graph.E * sizeof(int3) * 5);
	// cudaMalloc(&devTriEdgeSize, 1 * sizeof(int));
	// cudaMalloc(&devNewEdges, graph.E * sizeof(int3) * 5);
	// cudaMalloc(&devNewEdgeSize, 1 * sizeof(int));
	// cudaMalloc(&devNewEndEdges, graph.E * sizeof(int3) * 5);
	// cudaMalloc(&devNewEndEdgeSize, 1 * sizeof(int));


    if (graph.gw.Direction == EdgeType::DIRECTED) {
        cudaMalloc(&devInNodes, (graph.V + 1) * sizeof(int));
        cudaMalloc(&devInEdges, graph.E * sizeof(int2)*5);

		cudaMalloc(&devInLoads, graph.V * sizeof(int));
        cudaMalloc(&devInDones, graph.V * sizeof(int));
    }
	else
	{
		devInNodes = devOutNodes;
		devInEdges = devOutEdges;
		
		devInLoads = devOutNodes;
		devInDones = devOutDones;
	}

	__CUDA_ERROR("Graph Allocation");
}

cudaGraphPreDeal::~cudaGraphPreDeal() {
    cudaFree(devOutNodes);
    cudaFree(devOutEdges);

	cudaFree(devOutLoads);
    cudaFree(devOutDones);
	cudaFree(devMarks);
        
	cudaFree(devOrders);
	cudaFree(devCandidates);
	cudaFree(devCandidateSize);

	// cudaFree(devTriEdges);
	// cudaFree(devTriEdgeSize);
	// cudaFree(devNewEdges);
	// cudaFree(devNewEdgeSize);
	// cudaFree(devNewEndEdges);
	// cudaFree(devNewEndEdgeSize);

    if (graph.gw.Direction == EdgeType::DIRECTED) {
        cudaFree(devInNodes);
        cudaFree(devInEdges);

		cudaFree(devInLoads);
        cudaFree(devInDones);
    }
	else
	{
		devInNodes = NULL;
		devInEdges = NULL;
		
		devInLoads = NULL;
		devInDones = NULL;
	}
}

} //@cuda_graph
