#pragma once

#include <vector>
#include <map>
#include "GraphPreDeal.hpp"
#include "../../../Base/Host/BaseHost.hpp"
using std::vector;

namespace graph {
    class OverlayGraph{
	public:
    	vector<int> OutNodesVec;
        vector<int2> OutEdgesVec;
        void resize(int V)
        {
            OutNodesVec.resize(V+1);
        }
    };
}