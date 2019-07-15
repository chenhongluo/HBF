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
#include "../../include/Device/HBFGraph.cuh"
#include "Graph.hpp"
#include "XLib.hpp"

using namespace graph;

__device__ int devF2Size[4];

HBFGraph::HBFGraph(GraphSSSP& graph,
                   const bool _inverse_graph,
                   const int _degree_options) :
                   cudaGraphWeight(graph, _inverse_graph, _degree_options) {

	cudaMalloc(&devDistances, graph.V * sizeof (hdist_t));
    const int delta = (1 << 20) * 16;   //16 MB free

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    cudaMalloc(&devF1, (free - delta) / 2 );
    cudaMalloc(&devF2, (free - delta) / 2);
    __CUDA_ERROR("HBFGraph Allocation");

    max_frontier_size = ((free - delta) / 2) / sizeof(node_t);
    StreamModifier::thousandSep();
    std::cout << "Max frontier size: " <<  max_frontier_size << std::endl;
    StreamModifier::resetSep();

    devDistanceInit = new hdist_t[graph.V];
    std::fill(devDistanceInit, devDistanceInit + graph.V, INF);
}

HBFGraph::~HBFGraph() {
    cudaFree(devF1);
    cudaFree(devF2);
}

void HBFGraph::init(const node_t Sources[], int nof_sources) {
	cudaMemcpy(devF1, Sources, nof_sources * sizeof(node_t), cudaMemcpyHostToDevice);
	cudaMemcpy(devDistances, devDistanceInit, graph.V * sizeof (hdist_t), cudaMemcpyHostToDevice);
	if (nof_sources == 1){
        const hdist_t zero = ZERO;
		cudaMemcpy(devDistances + Sources[0], &zero, sizeof(hdist_t), cudaMemcpyHostToDevice);
	}
    int SizeArray[4] = {0, 0, 0, 0};
	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);
    __CUDA_ERROR("BellmanFord Kernel Init");

    global_sync::Reset();
}
