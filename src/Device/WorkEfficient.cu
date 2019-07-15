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

namespace
{
template <typename T>
inline bool distanceCompare(dist_t A, T B);

template <>
inline bool distanceCompare<dist_t>(dist_t A, dist_t B)
{
	return A == B;
}

template <>
inline bool distanceCompare<int2>(dist_t A, int2 B)
{
	return A == B.y;
}
} // namespace

void HBFGraph::FrontierDebug(const int FSize, const int level)
{
	if (FSize > max_frontier_size)
		__ERROR("Device memory not sufficient to contain the vertices frontier");
	if (CUDA_DEBUG)
	{
		//__CUDA_ERROR("BellmanFord Host");

		// std::cout << "level: " << level << "\tF2Size: " << FSize << std::endl;
		if (CUDA_DEBUG >= 2)
		{
			if (level <= DEBUG_LEVEL)
			{
				node_t *tmpF1 = new node_t[graph.V * 10];
				cudaMemcpy(tmpF1, devF1, FSize * sizeof(node_t), cudaMemcpyDeviceToHost);
				printf("\n%s=%d\t", "cuda_frontier_level:", level);
				printExt::host::printArray(tmpF1, FSize, " ");
				delete[] tmpF1;
			}
		}
	}
}

void HBFGraph::WorkEfficient()
{
	int SizeArray[4];
	long long int totalEdges = 0;
	float totalTime = 0;
	std::cout.setf(std::ios::fixed | std::ios::left);

	printf("make degree");

	this->markDegree();

	timer::Timer<timer::HOST> TM_H;
	timer_cuda::Timer<timer_cuda::DEVICE> TM_D;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<> distribution(0, graph.V);

	int maxFrontier = std::numeric_limits<int>::min();
	std::vector<int> host_frontiers;
	int *Sources = new int[1];
			if (CHECK_RESULT)
		{
					dynamic_cast<GraphSSSP &>(graph).BellmanFord_Queue_init();
		}
	for (int i = 0; i < N_OF_TESTS; i++)
	{

		Sources[0] = {N_OF_TESTS == 1 ? 0 : distribution(generator)};
		//Sources[0] = 160413;

		int edgeTraversed = graph.E;
		if (CHECK_TRAVERSED_EDGES)
		{
			graph.BFS_Init();
			graph.BFS(Sources[0]);
			edgeTraversed = graph.BFS_visitedEdges();
			graph.BFS_Reset();

			if (edgeTraversed == 0 || (float)graph.E / edgeTraversed < 0.1f)
			{
				i--;
				std::cout << "EdgeTraversed:" << edgeTraversed
						  << " -> Repeat" << std::endl;
				continue;
			}
		}

		//printf("host start\n");

		if (CHECK_RESULT)
		{
			//std::cout << "Computing Host Bellman-Ford..." << std::endl;
			if (CUDA_DEBUG)
			{
				//dynamic_cast<GraphSSSP&>(graph).BoostDijkstra(Sources[0]);
				dynamic_cast<GraphSSSP &>(graph).BellmanFord_Frontier(Sources[0], host_frontiers);
				printf("host_frontiers_size = ");
				for (int i = 0; i < (int)host_frontiers.size(); i++)
					std::cout << " " << host_frontiers[i];
				std::cout << std::endl;
				host_frontiers.resize(0);
			}
			else
			{
				dynamic_cast<GraphSSSP &>(graph).BellmanFord_Queue(Sources[0]);
			}
		}

		//printf("cuda start\n");

		int level = 1, F1Size = 1, F2Size;
		this->init(Sources);

		//======================================================================

		std::vector<int> cuda_frontiers;
		int *last_nodes = new int[this->graph.V];
		for (int i = 0; i < this->graph.V; i++)
			last_nodes[i] = i;
		TM_D.start();

		do
		{
			cuda_frontiers.push_back(F1Size);
			//printf("F1Size:%d",F1Size);
			FrontierDebug(F1Size, level);
			DynamicVirtualWarp(F1Size, level);

			cudaMemcpyFromSymbol(SizeArray, devF2Size, sizeof(int) * 4);
			F2Size = SizeArray[level & 3];
			F1Size = F2Size;

			level++;
			std::swap<int *>(devF1, devF2);
			maxFrontier = std::max(maxFrontier, F2Size);

		} while (F2Size > 0);
		// if (OUT_DEGREE_OPT)
		// {
		// 	printf("graph.V:%d\n", this->graph.V);

		// 	F1Size = this->graph.V;
		// 	cudaMemcpy(devF1, last_nodes, this->graph.V * sizeof(int), cudaMemcpyHostToDevice);
		// 	DynamicVirtualWarpForLast(F1Size, level);
		// 	//printf("delete\n");
		// }

		TM_D.stop();
		delete[] last_nodes;
		if (CUDA_DEBUG)
		{
			printf("cuda_frontiers_size = ");
			for (int x : cuda_frontiers)
			{
				printf(" %d", x);
			}
			printf("\n");
		}

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

		if (CHECK_RESULT)
		{
			dist_t *Dist = dynamic_cast<GraphSSSP &>(graph).BellmanFord_Result();
			if (CUDA_DEBUG >= 3)
			{
				printExt::host::printArray(Dist, graph.V, "host_distance:");
				int *devArray = new int[graph.V];
				cudaMemcpy(devArray, devDistances, graph.V * sizeof(int), cudaMemcpyDeviceToHost);
				printExt::host::printArray(devArray, graph.V, "cuda_distance:");
				delete[] devArray;
			}

			cuda_util::Compare(Dist, devDistances, graph.V, distanceCompare);
			printf("the %d test is ok", i);
					dynamic_cast<GraphSSSP &>(graph).BellmanFord_Queue_reset();
			//TODO chl
		}

		/*std::cout << "reset start" << std::endl;

	   std::cout << "reset end" << std::endl;*/
	}
	if (CHECK_RESULT)
		{
				dynamic_cast<GraphSSSP &>(graph).BellmanFord_Queue_end();
		}

	std::cout << std::endl
			  << "\tNumber of TESTS: " << N_OF_TESTS << std::endl
			  << "\t      Avg. Time: " << totalTime / N_OF_TESTS << " ms" << std::endl
			  << "\t     Avg. MTEPS: " << totalEdges / (totalTime * 1000) << std::endl
			  << "\t    maxFrontier: " << maxFrontier << std::endl
			  << std::endl;
}

inline void HBFGraph::DynamicVirtualWarpForLast(const int F1Size, const int level)
{
	int size = numeric::log2(RESIDENT_THREADS / F1Size);
	if (MIN_VW >= 1 && size < LOG2<MIN_VW>::value)
		size = LOG2<MIN_VW>::value;
	if (MAX_VW >= 1 && size > LOG2<MAX_VW>::value)
		size = LOG2<MAX_VW>::value;

#define funB(a) kernels::chl_kernel<(a), false>         \
	<<<_Div(graph.V, (BLOCKDIM / (a)) * ITEM_PER_WARP), \
	   BLOCKDIM,                                        \
	   SMem_Per_Block<char, BLOCKDIM>::value>>>(devOutNodes, devOutEdges, devDistances, devF1, devF2, F1Size, level);

	def_SWITCHB(size);
#undef funB
}

inline void HBFGraph::DynamicVirtualWarp(const int F1Size, const int level)
{
	int size = numeric::log2(RESIDENT_THREADS / F1Size);
	if (MIN_VW >= 1 && size < LOG2<MIN_VW>::value)
		size = LOG2<MIN_VW>::value;
	if (MAX_VW >= 1 && size > LOG2<MAX_VW>::value)
		size = LOG2<MAX_VW>::value;
//printf("VW_SIZE:%d\n", size);

/*//#define fun(a)	BF_Kernel1<(a), false>\
    //                    <<<std::min(_DIV(graph.V, BLOCKDIM), 96), BLOCKDIM, SM_DYN>>>\
    //					(devOutNode, devOutEdge, devDistance, devF1, devF2,  F1Size, level);*/

//TODO --chl
#define fun(a) kernels::BF_Kernel1<(a), false>          \
	<<<_Div(graph.V, (BLOCKDIM / (a)) * ITEM_PER_WARP), \
	   BLOCKDIM,                                        \
	   SMem_Per_Block<char, BLOCKDIM>::value>>>(devOutNodes, devOutEdges, devDistances, devF1, devF2, F1Size, level);

	def_SWITCH(size);

#undef fun
}
