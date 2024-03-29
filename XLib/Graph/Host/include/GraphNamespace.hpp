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
#include <Base/Base.hpp>
#include <vector_types.h>
#include <vector_functions.hpp>

namespace graph {
	enum class	    EdgeType { DIRECTED, UNDIRECTED, UNDEF_EDGE_TYPE };
	enum class     GraphType { NORMAL, MULTIGRAPH, UNDEF_GRAPH_TYPE  };
    enum class AttributeType { BINARY, INTEGER, REAL, SIGN           };

    using node_t   = int;
    using edge_t   = int;
	using node_t2  = node_t[2];
    using degree_t = int;
	using dist_t   = int;
    using weight_t = int;

    // struct int2{int x, int y};
    // struct int3{int x, int y, int z};
    // struct int4{int x, int y, int z, int w};
    //for vs code

    void readHeader(const char* File, int &V, int &E, int &nof_lines);
    void readHeader(const char* File, int &V, int &E, int &nof_lines, EdgeType& edgeType);
    void readHeader(const char* File, int &V, int &E, int &nof_lines, EdgeType& edgeType, GraphType& graphType);
    void readHeader(const char* File, int &V, int &E, int &nof_lines, EdgeType& edgeType, AttributeType& attributeType);
}
