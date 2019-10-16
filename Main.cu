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

#include <../config.cuh>
#include <XLib.hpp>
#include <Host/GraphSSSP.hpp>
#include <Device/cudaOverlayGraph.cuh>
using namespace cuda_graph;

int main(int argc, char** argv) {
	if (argc < 2)
		__ERROR("No Input File");

	cuda_util::cudaStatics();

    node_t V; edge_t E; int nof_lines;
    graph::EdgeType edgeType = graph::EdgeType::UNDEF_EDGE_TYPE;
    graph::readHeader(argv[1], V, E, nof_lines, edgeType);

    GraphSSSP graph(V, E, edgeType);
    graph.read(argv[1], nof_lines);

    GraphWeight& gw=graph;
    printf("init PreDeal\n");
    GraphPreDeal gp(gw);
    printf("start PreDeal\n");
    PreDealGraph(gp);
    printf("end PreDeal\n");
    cudaGNRGraph gnr(gp,ALL_LEVELS,upSearchStrategy,downSearchStrategy);
    printf("\nmalloc devgraph");
    gnr.cudaMallocMem();
    __CUDA_ERROR("malloc");
    printf("\ncopy to device");
    gnr.cudaCopyMem();
    __CUDA_ERROR("copy");
    gnr.WorkEfficient(gw);

   // graph.DijkstraSET(0);

#if defined(BOOST_FOUND)
    graph.BoostDijkstra(0);
    graph.BoostBellmanFord(0);
#endif
}
