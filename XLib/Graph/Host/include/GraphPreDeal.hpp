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
