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
#include "Device/HBFGraph.cuh"
#include "XLib.hpp"

#include "Kernels/WorkEfficient_KernelDispath.cu"

namespace {
    template<typename T>
    inline bool distanceCompare(dist_t A, T B);

    template<>
    inline bool distanceCompare<dist_t>(dist_t A, dist_t B) {
        return A == B;
    }

    template<>
    inline bool distanceCompare<int2>(dist_t A, int2 B) {
        return A == B.y;
    }
}

void HBFGraph::WorkEfficient() {
	int SizeArray[4];
	long long int totalEdges = 0;
	float totalTime = 0;
	std::cout.setf(std::ios::fixed | std::ios::left);

	this->markDegree();

    timer::Timer<timer::HOST> TM_H;
	timer_cuda::Timer<timer_cuda::DEVICE> TM_D;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<> distribution(0, graph.V);

    if (CHECK_TRAVERSED_EDGES)
        graph.BFS_Init();
    if (CHECK_RESULT)
        dynamic_cast<GraphSSSP&>(graph).BellmanFord_Queue_init();

    int maxFrontier = std::numeric_limits<int>::min();
    std::vector<int> host_frontiers;

	for (int i = 0; i < N_OF_TESTS; i++) {

		const int Sources[] = { N_OF_TESTS == 1 ? 0 : distribution(generator) };

		int edgeTraversed = graph.E;
		if (CHECK_TRAVERSED_EDGES) {
            graph.BFS(Sources[0]);
            edgeTraversed = graph.BFS_visitedEdges();
            graph.BFS_Reset();

			if (edgeTraversed == 0 || (float) graph.E / edgeTraversed < 0.1f) {
				i--;
				std::cout << "EdgeTraversed:" << edgeTraversed
                          << " -> Repeat" << std::endl;
				continue;
			}
		}
		if (CHECK_RESULT) {
			std::cout << "Computing Host Bellman-Ford..." << std::endl;
            if (CUDA_DEBUG) {
                dynamic_cast<GraphSSSP&>(graph).BellmanFord_Frontier(Sources[0], host_frontiers);
                for (int i = 0; i < (int) host_frontiers.size(); i++)
                    std::cout << i << "\t" << host_frontiers[i] << std::endl;
                host_frontiers.resize(0);
            }
            else
			     dynamic_cast<GraphSSSP&>(graph).BellmanFord_Queue(Sources[0]);
		}
		int level = 1, F1Size = 1, F2Size;
		this->init(Sources);

		//======================================================================

		TM_D.start();

		do {
			DynamicVirtualWarp(F1Size, level);

			cudaMemcpyFromSymbol(SizeArray, devF2Size, sizeof(int) * 4);
			F2Size = SizeArray[level & 3];
			F1Size = F2Size;
			level++;
			FrontierDebug(F2Size, level);
			std::swap<int*>(devF1, devF2);
			maxFrontier = std::max(maxFrontier, F2Size);

		} while ( F2Size > 0 );

		TM_D.stop();

		//======================================================================

		float time = TM_D.duration();
		totalTime += time;

		__CUDA_ERROR("BellmanFord Kernel");

		totalEdges += edgeTraversed;
		if (N_OF_TESTS > 1)
			std::cout << "iter: " << std::setw(10) << i
                      << "\ttime: " << std::setw(10) << time
                      << "\tEdges: " << std::setw(10) << edgeTraversed
                      << "\tsource: " << Sources[0] << std::endl;

        if (CHECK_RESULT) {
            dist_t* Dist = dynamic_cast<GraphSSSP&>(graph).BellmanFord_Result();
    	    cuda_util::Compare(Dist, devDistances, graph.V, distanceCompare);
       }
       dynamic_cast<GraphSSSP&>(graph).BellmanFord_Queue_reset();
	}

	std::cout << std::endl
              << "\tNumber of TESTS: " << N_OF_TESTS << std::endl
			  << "\t      Avg. Time: " << totalTime / N_OF_TESTS << " ms" << std::endl
			  << "\t     Avg. MTEPS: " << totalEdges / (totalTime * 1000) << std::endl
			  << "\t    maxFrontier: " << maxFrontier << std::endl << std::endl;
}

void HBFGraph::FrontierDebug(const int FSize, const int level) {
    if (FSize > max_frontier_size)
        __ERROR("Device memory not sufficient to contain the vertices frontier");
	if (CUDA_DEBUG) {
		__CUDA_ERROR("BellmanFord Host");

		std::cout << "level: " << level << "\tF2Size: " << FSize << std::endl;
		if (CUDA_DEBUG >= 2) {
            node_t* tmpF1 = new node_t[graph.V];
			cudaMemcpy(tmpF1, devF1, FSize * sizeof(node_t), cudaMemcpyDeviceToHost);
			printExt::device::printArray(tmpF1, FSize, "Frontier:\t");
		}
	}
}

inline void HBFGraph::DynamicVirtualWarp(const int F1Size, const int level) {
    int size = numeric::log2(RESIDENT_THREADS / F1Size);
	if (MIN_VW >= 1 && size < LOG2<MIN_VW>::value)
		size = LOG2<MIN_VW>::value;
	if (MAX_VW >= 1 && size > LOG2<MAX_VW>::value)
		size = LOG2<MAX_VW>::value;

    /*//#define fun(a)	BF_Kernel1<(a), false>\
    //                    <<<std::min(_DIV(graph.V, BLOCKDIM), 96), BLOCKDIM, SM_DYN>>>\
    //					(devOutNode, devOutEdge, devDistance, devF1, devF2,  F1Size, level);*/

    #define fun(a)  kernels::BF_Kernel1<(a), false>                             \
                        <<< _Div(graph.V, (BLOCKDIM / (a)) * ITEM_PER_WARP),    \
                        BLOCKDIM,                                               \
                        SMem_Per_Block<char, BLOCKDIM>::value >>>               \
                        (devOutNodes, devOutEdges, devDistances, devF1, devF2,  F1Size, level);

    def_SWITCH(size);

    #undef fun
}
