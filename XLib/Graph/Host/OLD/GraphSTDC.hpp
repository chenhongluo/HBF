#pragma once

#include <vector_types.h>	//int2
#include "GraphSTD.hpp"

namespace graph {

	class GraphSTDC : public GraphSTD {
		public:
			node_t2 *OutNodes2, *InNodes2;

			GraphSTD(const int _V, const int _E, const GDirection _Direction);
	};
}
