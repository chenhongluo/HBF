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
namespace kernels {

	template<int VW_SIZE, bool SYNC>
	__GLOBAL_DEVICE__ void NAME0(edge_t* __restrict__  devNodes,
		int2* __restrict__    devEdges,
		hdist_t* __restrict__ devDistances,
		node_t* __restrict__  devF1,
		node_t* __restrict__  devF2,
		const int devF1Size,
		const dist_t level) {
		const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / VW_SIZE;
		const int Stride = gridDim.x * (BLOCKDIM / VW_SIZE);
		for (int t = VirtualID; t < devF1Size; t += Stride) {
			const node_t index = cub::ThreadLoad<cub::LOAD_CS>(devF1 + t);

			const weight_t nodeWeight = cub::ThreadLoad<cub::LOAD_CS>(reinterpret_cast<dist_t*>(devDistances) + (ATOMIC64 ? (index << 1) | 1 : index));//<<1 == *2; 
			const edge_t start = cub::ThreadLoad<cub::LOAD_CS>(devNodes + index);
			edge_t end = cub::ThreadLoad<cub::LOAD_CS>(devNodes + index + 1);
			for (int k = start + (threadIdx.x & MOD2<VW_SIZE>::value); k < end; k += VW_SIZE) {
				const int2 dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdges + k);		// X: edge index	Y: edge weight
				if (nodeWeight != INT_MAX && dest.x & 0x40000000)
				{
					const weight_t newWeight = nodeWeight + dest.y;
#if ATOMIC64
					//TODO--chl level--newWeight?
					int2 toWrite = make_int2(level, newWeight);
					atomicMin(reinterpret_cast<unsigned long long*>(&devDistances[dest.x & 0x3FFFFFFF]),
						reinterpret_cast<unsigned long long&>(toWrite));
#else
					atomicMin(&devDistances[dest.x & 0x3FFFFFFF], newWeight);
#endif
				}
			}
		}
	}

	template<int VW_SIZE, bool SYNC>
	__GLOBAL_DEVICE__ void NAME1(edge_t* __restrict__  devNodes,
		int2* __restrict__    devEdges,
		hdist_t* __restrict__ devDistances,
		node_t* __restrict__  devF1,
		node_t* __restrict__  devF2,
		const int devF1Size,
		const dist_t level) {

		node_t* devF2SizePrt = &devF2Size[level & 3];	// mod 4
		//assert(*devF2SizePrt == 0);
		if (blockIdx.x == 0 && threadIdx.x == 0)
			devF2Size[(level + 1) & 3] = 0;

		// if (blockIdx.x == 0 && threadIdx.x == 0)
		// {
		// 	int a = 4;
		// 	int b = 10;
		// 	int2 c = make_int2(a,b);
		// 	printf("test:%lld,%llx\n",reinterpret_cast<unsigned long long&>(c),reinterpret_cast<unsigned long long&>(c));
		// }

		int founds = 0;
		node_t Queue[REG_LIMIT];

		const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / VW_SIZE;
		const int Stride = gridDim.x * (BLOCKDIM / VW_SIZE);

		if (!SAFE) {
			for (int t = VirtualID; t < devF1Size; t += Stride) {
				const node_t index = cub::ThreadLoad<cub::LOAD_CS>(devF1 + t);

				const weight_t nodeWeight = cub::ThreadLoad<cub::LOAD_CS>(reinterpret_cast<dist_t*>(devDistances) + (ATOMIC64 ? (index << 1) | 1 : index));//<<1 == *2; 
				const edge_t start = cub::ThreadLoad<cub::LOAD_CS>(devNodes + index);
				edge_t end = cub::ThreadLoad<cub::LOAD_CS>(devNodes + index + 1);

				DynamicParallelism<VW_SIZE>(devEdges, devDistances, devF2, index, nodeWeight, start, end, level);
				EdgeVisit<VW_SIZE>(devEdges, devDistances, devF2, devF2SizePrt, index, nodeWeight, start, end, Queue, founds, level);
			}
		}
		else {
			const int size = ceilf(__fdividef(devF1Size, gridDim.x));
			const int maxLoop = (size + BLOCKDIM / VW_SIZE - 1) >> (LOG2<BLOCKDIM>::value - LOG2<VW_SIZE>::value);

			for (int t = VirtualID, loop = 0; loop < maxLoop; t += Stride, loop++) {
				node_t index;
				weight_t nodeWeight;
				edge_t start, end = INT_MIN;
				if (t < devF1Size) {
					index = cub::ThreadLoad<cub::LOAD_CS>(devF1 + t);
					/*if (blockIdx.x == 0 && threadIdx.x < 128)
					{
						printf("threadIdx.x = %d,VWId = %d,index = %d\n", threadIdx.x, VirtualID, index);
					}
*/
					nodeWeight = cub::ThreadLoad<cub::LOAD_CS>(reinterpret_cast<int*>(devDistances) + (ATOMIC64 ? (index << 1) | 1 : index));
					start = cub::ThreadLoad<cub::LOAD_CS>(devNodes + index);
					end = cub::ThreadLoad<cub::LOAD_CS>(devNodes + index + 1);
					// if ((blockIdx.x * BLOCKDIM + threadIdx.x) % VW_SIZE == 0 && end-start > 0)
					// {
					// 	printf("VW_Id:%d,index:%d,load:%d\n", VirtualID, index, end-start);
					// }

					DynamicParallelism<VW_SIZE>(devEdges, devDistances, devF2, index, nodeWeight, start, end, level);
				}
				// chen
				EdgeVisit<VW_SIZE>(devEdges, devDistances, devF2, devF2SizePrt, index, nodeWeight, start, end, Queue, founds, level);
			}
		}
		//if (blockIdx.x == 0 && threadIdx.x <= 3)
		//{
		//	printf("threadIdx.x = %d,%d,%d\n", threadIdx.x,founds, devF2Size[level & 3]);
		//}
		int prefix_sum = founds;
		const int warp_offset = WarpExclusiveScan<>::AddAtom(prefix_sum, devF2SizePrt);
		// if ((blockIdx.x * BLOCKDIM + threadIdx.x) % VW_SIZE == 0 && prefix_sum > 0 )
		// {
		// 	printf("VW_Id:%d,founds:%d\n", VirtualID, warp_offset);
		// }

		// if (blockIdx.x == 0 && threadIdx.x < 128)
		// {
			//printf("Main::threadIdx.x = %d,%d,%d\n ", threadIdx.x, warp_offset, devF2Size[level & 3]);
			// for (int i = 0; i < founds; i++)
			// {
			// 	printf("threadIdx.x = %d,queue = %d ", threadIdx.x, Queue[i]);
			// }
		// }
		// else
		// {
			
		// }
		
		thread::RegToGlobal(Queue, founds, devF2 + warp_offset + prefix_sum);
	}

} //@kernels
