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

#include <cub/cub.cuh>
#include "XLib.hpp"

namespace kernels {

__device__ __forceinline__ void KRelax(	const int2 dest,
                                        const node_t index,
                                        const weight_t nodeWeight,
										hdist_t* devDistance,
                                        node_t* Queue,
                                        int& founds,
                                        const int level) {

	if (OUT_DEGREE_OPT && (dest.x & 0x40000000))
		;
	else if (RING && (IN_DEGREE_OPT ? dest.x & 0x3FFFFFFF : dest.x) == index)
		;
	else if (level == 1) {
		const node_t destTMP = IN_DEGREE_OPT ? dest.x & 0x3FFFFFFF : dest.x;
		if (destTMP != index) {
			devDistance[ destTMP ] = MAKE_DIST(1, dest.y);
			Queue[founds++] = destTMP;
		}
	}
	else if (IN_DEGREE_OPT && (dest.x & 0x80000000)) {
		const weight_t newWeight = nodeWeight + dest.y;
		const node_t destTMP = dest.x & 0x3FFFFFFF;
		devDistance[ destTMP ] = MAKE_DIST(level, newWeight);
		Queue[founds++] = destTMP;
	}
	else {
		const weight_t newWeight = nodeWeight + dest.y;
	#if ATOMIC64
		int2 toWrite = make_int2(level, newWeight);
		unsigned long long aa =	atomicMin(	reinterpret_cast<unsigned long long*>(&devDistance[ dest.x ]),
											reinterpret_cast<unsigned long long&>(toWrite));
		const int2 toTest = reinterpret_cast<int2&>( aa );
		if (toTest.x != level && toTest.y > newWeight) {
	#else
		if (atomicMin(&devDistance[ dest.x ], newWeight) > newWeight) {
	#endif
				Queue[founds++] = dest.x;
		}
	}

}

using namespace data_movement::dynamic;

template<int VW_SIZE, int SIZE>
__device__ __forceinline__ void EdgeVisit(	int2* __restrict__ devEdges,
                                            hdist_t* __restrict__ devDistances,
											node_t* __restrict__ devF2,
                                            int* __restrict__ devF2SizePrt,
											const node_t index,
                                            const weight_t nodeWeight,
                                            const edge_t start,
                                            edge_t end,
											node_t (&Queue)[SIZE],
                                            int& founds,
                                            const int level) {
if (!SAFE)
	for (int k = start + (threadIdx.x & MOD2<VW_SIZE>::value); k < end; k += VW_SIZE) {
		const int2 dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdges + k);		// X: edge index	Y: edge weight

		KRelax(dest, index, nodeWeight, devDistances, Queue, founds, level);
	}
else {
	bool flag = true;
	edge_t k = start + (threadIdx.x & MOD2<VW_SIZE>::value);
	while (flag) {
		while (k < end && founds < REG_LIMIT) {
			const int2 dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdges + k);		// X: edge index	Y: edge weight

			KRelax(dest, index, nodeWeight, devDistances, Queue, founds, level);
			k += VW_SIZE;
		}
		if (__any(founds >= REG_LIMIT)) {
            int prefix_sum = founds;
            const int warp_offset = WarpExclusiveScan<>::AddAtom(prefix_sum, devF2SizePrt);
            thread::RegToGlobal(Queue, founds, devF2 + warp_offset + prefix_sum);
			founds = 0;
		} else
			flag = false;
	}
}
}

//------------------------------------------------------------------------------

__global__ void DynamicKernel( 	int2* __restrict__ devEdge,
								hdist_t* __restrict__ devDistance,
                                node_t* devF2,
								const node_t index,
                                const weight_t nodeWeight,
                                const edge_t start,
                                const edge_t end,
								const int level) {

	int* devF2SizePrt = &devF2Size[level & 3];
	const int ID = blockIdx.x * BLOCKDIM + threadIdx.x;

	node_t Queue[REG_LIMIT];
	int founds = 0;
	for (int k = start + ID; k < end; k += gridDim.x * BLOCKDIM) {
		const int2 dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdge + k);		// X: edge index	Y: edge weight

		KRelax(dest, index, nodeWeight, devDistance, Queue, founds, level);
	}
    int prefix_sum = founds;
    const int warp_offset = WarpExclusiveScan<>::AddAtom(prefix_sum, devF2SizePrt);
    thread::RegToGlobal(Queue, founds, devF2 + warp_offset + prefix_sum);
}


template<int VW_SIZE>
__device__ __forceinline__ void DynamicParallelism( int2* __restrict__ devEdges,
                                                    hdist_t* __restrict__ devDistances,
                                                    node_t* devF2,
													const node_t index,
                                                    const weight_t nodeWeight,
                                                    const edge_t start,
                                                    edge_t& end,
													const int level) {
	if (DYNAMIC_PARALLELISM) {
		const int degree = end - start;
		if (degree >= THRESHOLD_G) {
			if ((threadIdx.x & MOD2<VW_SIZE>::value) == 0) {
				const int DynGridDim = (degree + (BLOCKDIM * EDGE_PER_TH) - 1) >> (LOG2<BLOCKDIM>::value + LOG2<EDGE_PER_TH>::value);
				DynamicKernel<<< DynGridDim, BLOCKDIM>>> (devEdges, devDistances, devF2, index, nodeWeight, start, end, level);
			}
			end = INT_MIN;
		}
	}
}

} //@kernels
