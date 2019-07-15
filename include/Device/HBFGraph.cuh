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

#include "../../config.cuh"
#include "Host/GraphSSSP.hpp"
#include "Graph.hpp"
using namespace cuda_graph;

extern __device__ int devF2Size[4];
//extern __device__ int devMinL;
//extern __device__ int devMinM;


extern  int SMem[];

class HBFGraph : public cudaGraphWeight {
private:
    node_t *devF1, *devF2;
    hdist_t* devDistances;
    hdist_t* devDistanceInit;
    int max_frontier_size;

    inline void DynamicVirtualWarp(const int F1Size, const int level);
	inline void DynamicVirtualWarpForLast(const int F1Size, const int level);
    void FrontierDebug(const int FSize, const int level);
    void markDegree();

public:
    HBFGraph(GraphSSSP& graph,
             const bool _inverse_graph,
             const int _degree_options = 0);

    ~HBFGraph();

    void init(const node_t Sources[], int nof_sources = 1);

    void WorkEfficient();
};
