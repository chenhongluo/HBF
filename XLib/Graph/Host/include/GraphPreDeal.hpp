
#pragma once

#include <vector>
#include "GraphWeight.hpp"
#include "../../../Base/Host/BaseHost.hpp"
using std::vector;

namespace graph {
    static void ToCSR(vector<int3>& TriEdgesVec,
    vector<int>& Nodes,
    vector<int2>& Edges,
    int k);

    // static void CSRMerge(vector<int> &Nodes1,
	// vector<int2>& Edges1,
	// vector<int> &Nodes2,
	// vector<int2>& Edges2
    // );

    void CSRMerge(vector<int3>& TriEdgesVec,vector<int>& Nodes,
    vector<int2>& Edges,int k);

	class GraphPreDeal{
	public:
        int V;
        int E;
        const GraphWeight& gw;
        vector<int> InNodesVec;
        vector<int2> InEdgesVec;
        vector<int> OutNodesVec;
        vector<int2> OutEdgesVec;

        vector<int> Orders;
        vector<int> Candidates;
        vector<float> Marks;
        vector<int> ContractedNodes;
        vector<int3> ContractedEdges;
        vector<int3> AddEdges;

		GraphPreDeal(const GraphWeight& gw);
        ~GraphPreDeal();
	};
}
