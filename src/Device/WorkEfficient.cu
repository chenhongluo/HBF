#include <Device/cudaOverlayGraph.cuh>
#include <../config.cuh>
//#include "Kernels/WorkEfficient_KernelDispath.cu"

#include <vector>
#include<random>

using std::vector;
// void cudaGNRGraph::FrontierDebug(const int FSize, const int level)
// {
// 	if (FSize > max_frontier_size)
// 		__ERROR("Device memory not sufficient to contain the vertices frontier");
// 	if (CUDA_DEBUG)
// 	{
// 		//__CUDA_ERROR("BellmanFord Host");

// 		// std::cout << "level: " << level << "\tF2Size: " << FSize << std::endl;
// 		if (CUDA_DEBUG >= 2)
// 		{
// 			if (level <= DEBUG_LEVEL)
// 			{
// 				node_t *tmpF1 = new node_t[graph.V * 10];
// 				cudaMemcpy(tmpF1, devF1, FSize * sizeof(node_t), cudaMemcpyDeviceToHost);
// 				printf("\n%s=%d\t", "cuda_frontier_level:", level);
// 				printExt::host::printArray(tmpF1, FSize, " ");
// 				delete[] tmpF1;
// 			}
// 		}
// 	}
// }
namespace cuda_graph {
void cudaGNRGraph::WorkEfficient(GraphWeight& graph)
{
	//GraphSSSP& graph = (GraphSSSP&)gw;
	long long int totalEdges = 0;
	float totalTime = 0;
	std::cout.setf(std::ios::fixed | std::ios::left);

	timer::Timer<timer::HOST> TM_H;


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
	int needCount =0;
	for (int i = 0; i < N_OF_TESTS; i++)
	{
		// Sources[0] = {N_OF_TESTS == 1 ? 0 : distribution(generator)};
		//Sources[0] = 160413;
		Sources[0] = TEST_NODES[i];
		// printfInt(gp.Orders[Sources[0]],"source_order");
		// if(gp.Orders[TEST_NODES[i]]>10 || gp.Orders[TEST_NODES[i]]<5)
		// {
		// 	continue;
		// }
		// if(needCount >= 10)
		// 	break;
		// needCount++;
		
		
		timer_cuda::Timer<timer_cuda::DEVICE> TM_D;
		int edgeTraversed = graph.E;
		// if (CHECK_TRAVERSED_EDGES)
		// {
		// 	graph.BFS_Init();
		// 	graph.BFS(Sources[0]);
		// 	edgeTraversed = graph.BFS_visitedEdges();
		// 	graph.BFS_Reset();

		// 	if (edgeTraversed == 0 || (float)graph.E / edgeTraversed < 0.1f)
		// 	{
		// 		i--;
		// 		std::cout << "EdgeTraversed:" << edgeTraversed
		// 				  << " -> Repeat" << std::endl;
		// 		continue;
		// 	}
		// }

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

		printf("cuda start\n");
		printf("source id:%d\n",Sources[0]);

		TM_D.start();
		GNRSearchMain(Sources[0]);
		__CUDA_ERROR("BellmanFord Kernel");

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

		if (CHECK_RESULT)
		{
			// printf("the %d test is ok", i);
			dist_t *Dist = dynamic_cast<GraphSSSP &>(graph).BellmanFord_Result();
			// if (CUDA_DEBUG >= 3)
			// {
			// 	printExt::host::printArray(Dist, graph.V, "host_distance:");
			// 	printExt::host::printArray(devArray, graph.V, "cuda_distance:");
			// 	delete[] devArray;
			// }
			vector<int> devDist(V);
			copyDistance(devDist);
			//printf("the %d test is ok", i);
			int count_error=0;
			for(int j=0;j<V;j++)
			{
				if(devDist[j] != Dist[j])
				{
					count_error++;	// exit(-1);
					// printf("%d not equal,dev:%d,host:%d,cha:%d\n",j,devDist[j],Dist[j],devDist[j]-Dist[j]);
				}
			}
			printf("the %d test is %f", i,(float)count_error/(float)V);
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
}
