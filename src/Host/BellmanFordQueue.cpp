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
#include "Host/GraphSSSP.hpp"
#include "../../config.cuh"
#include <queue>

void GraphSSSP::BellmanFord_Queue_init() {
	Distance = new dist_t[V];
    BellmanFord_Queue_reset();
	//printExt::host::printArray(Distance, V, "init dis:");
}

void GraphSSSP::BellmanFord_Queue_end() {
	delete[] Distance;
}

void GraphSSSP::BellmanFord_Queue_reset() {
    std::fill(Distance, Distance + V, std::numeric_limits<dist_t>::max());
}

dist_t* GraphSSSP::BellmanFord_Result() {
    return Distance;
}

void GraphSSSP::BellmanFord_Queue(const node_t source) {
	std::queue<node_t> Queue;
	Queue.push(source);

	Distance[source] = 0;

	while (Queue.size() > 0) {
		const node_t next = Queue.front();
		Queue.pop();
		for (int i = OutNodes[next]; i < OutNodes[next + 1]; i++) {
			const node_t dest = OutEdges[i];
			if (Relax(next, dest, Weights[i], Distance))
				Queue.push(dest);
		}
	}
}

void GraphSSSP::BellmanFord_Frontier(const node_t source, std::vector<int>& Frontiers) {
	std::queue<node_t> Queue1;
	std::queue<node_t> Queue2;
	Queue1.push(source);

	Distance[source] = 0;

	int level = 0, visited = 0;
	while (Queue1.size() > 0) {
		if (CUDA_DEBUG >= 2)
		{
			if (level < DEBUG_LEVEL)
			{
				printf("%s=%d\t", "host_frontier_level:", level);
				printExt::host::printQueue<node_t>(Queue1, " ");
				//printExt::host::printArray(Distance, V, "test_dis:");
			}

		}


		while (Queue1.size() > 0) {
			const node_t next = Queue1.front();
			Queue1.pop();
			visited++;
			for (int i = OutNodes[next]; i < OutNodes[next + 1]; i++) {
				const node_t dest = OutEdges[i];
                /*if (level == 1)
                    std::cout << "(" << next << "," << dest << ") -> " << Weights[i] << std::endl;*/
				if (Relax(next, dest, Weights[i], Distance))
					Queue2.push(dest);
			}
		}

		if (Queue2.size() > 0) {
			level++;
			Frontiers.push_back(Queue2.size());
			//chl TODO
			if (CUDA_DEBUG >= 2)
			{
				if (level < DEBUG_LEVEL)
				{
					printf("%s=%d\t", "host_frontier_level:", level);
					printExt::host::printQueue<node_t>(Queue2, " ");
					//printExt::host::printArray(Distance, V, "test_dis:");
				}
			}
		}

		while (Queue2.size() > 0) {
			const node_t next = Queue2.front();
			Queue2.pop();
			visited++;
			for (int i = OutNodes[next]; i < OutNodes[next + 1]; i++) {
				const node_t dest = OutEdges[i];
                /*if (level == 1)
                    std::cout << "(" << next << "," << dest << ") -> " << Weights[i] << std::endl;*/
				if (Relax(next, dest, Weights[i], Distance))
					Queue1.push(dest);
			}
		}
		if (Queue1.size() > 0)
		{
			level++;
			Frontiers.push_back(Queue1.size());
		}
	}
	//printf("%s=%d\t", "host frontier & level_all:", level);
}


bool GraphSSSP::Relax(const node_t u, const node_t v,
                      const weight_t weight, dist_t* Distance) {

	if (Distance[u] + weight < Distance[v]) {
		Distance[v] = Distance[u] + weight;
		return true;
	}
	return false;
}
