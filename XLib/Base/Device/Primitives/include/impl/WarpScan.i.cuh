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
#include "../../../../Host/BaseHost.hpp"
#include "../../../Util/Util.cuh"
#include <cub/cub.cuh>

namespace primitives {

	using namespace numeric;
	using namespace PTX;

	namespace {

#define warpInclusiveScan(ASM_OP, ASM_T, ASM_CL)                               \
    const int MASK = ((-1 << LOG2<WARP_SZ>::value) & 31) << 8;                 \
    _Pragma("unroll")	                                                       \
    for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {                 \
    	asm(                                                                   \
    		"{"                                                                \
            ".reg ."#ASM_T" r1;"                                               \
    		".reg .pred p;"                                                    \
    		"shfl.up.b32 r1|p, %1, %2, %3;"                                    \
            "@p "#ASM_OP"."#ASM_T" r1, r1, %1;"                                \
            "mov."#ASM_T" %0, r1;"                                             \
            "}"                                                                \
    		: "="#ASM_CL(value) : #ASM_CL(value), "r"(STEP), "r"(MASK));       \
    }

		//==============================================================================

		template<int WARP_SZ, bool BROADCAST, typename T>
		struct WarpInclusiveScanHelper {
			static __device__ __forceinline__ void Add(T& value);
		};

		template<int WARP_SZ, bool BROADCAST>
		struct WarpInclusiveScanHelper<WARP_SZ, BROADCAST, int> {
			static __device__ __forceinline__ void Add(int& value) {
				warpInclusiveScan(add, s32, r)
			}
		};

		template<int WARP_SZ, bool BROADCAST>
		struct WarpInclusiveScanHelper<WARP_SZ, BROADCAST, float> {
			static __device__ __forceinline__ void Add(float& value) {
				warpInclusiveScan(add, f32, f)
			}
		};
	} //@anonymous
#undef warpInclusiveScan
//==============================================================================

	template<int WARP_SZ>
	template<typename T>
	__device__ __forceinline__
		void WarpInclusiveScan<WARP_SZ>::Add(T& value) {
		WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
	}

	template<int WARP_SZ>
	template<typename T>
	__device__ __forceinline__
		void WarpInclusiveScan<WARP_SZ>::Add(T& value, T& total) {

		WarpInclusiveScanHelper<WARP_SZ, true, T>::Add(value);
		total = __shfl(value, WARP_SZ - 1, WARP_SZ);
	}

	template<int WARP_SZ>
	template<typename T>
	__device__ __forceinline__
		void WarpInclusiveScan<WARP_SZ>::Add(T& value, T* pointer) {

		WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
		if (LaneID() == WARP_SZ - 1)
			*pointer = value;
	}

	//==============================================================================

	template<int WARP_SZ>
	template<typename T>
	__device__ __forceinline__ void WarpExclusiveScan<WARP_SZ>::Add(T& value) {
		WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
		const int MASK = ((-1 << LOG2<WARP_SZ>::value) & 31) << 8;
		asm(
			"{"
			".reg .pred p;"
			"shfl.up.b32 %0|p, %1, %2, %3;"
			"@!p mov.b32 %0, 0;"
			"}"
			: "=r"(value) : "r"(value), "r"(1), "r"(MASK));
	}


	template<int WARP_SZ>
	template<typename T>
	__device__ __forceinline__
		void WarpExclusiveScan<WARP_SZ>::Add(T& value, T& total) {

		WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
		// if (blockIdx.x == 0 && threadIdx.x < 32)
		// {
		// 	printf("warp scan helper threadIdx.x = %d,%d,%d\n", threadIdx.x, value, total);
		// }
		const int MASK = ((-1 << LOG2<WARP_SZ>::value) & 31) << 8;
		total = value;
		asm(
			"{"
			".reg .pred p;"
			"shfl.up.b32 %0|p, %1, %2, %3;"
			"@!p mov.b32 %0, 0;"
			"}"
			: "=r"(value) : "r"(value), "r"(1), "r"(MASK));
	}

	template<int WARP_SZ>
	template<typename T>
	__device__ __forceinline__
		void WarpExclusiveScan<WARP_SZ>::AddBcast(T& value, T& total) {

		WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
		total = __shfl(value, WARP_SZ - 1, WARP_SZ);
		const int MASK = ((-1 << LOG2<WARP_SZ>::value) & 31) << 8;
		asm(
			"{"
			".reg .pred p;"
			"shfl.up.b32 %0|p, %1, %2, %3;"
			"@!p mov.b32 %0, 0;"
			"}"
			: "=r"(value) : "r"(value), "r"(1), "r"(MASK));
	}

	//------------------------------------------------------------------------------

	//!!!!!!!!! to improve

	template<int WARP_SZ>
	template<typename T>
	__device__ __forceinline__
		void WarpExclusiveScan<WARP_SZ>::Add(T& value, T* pointer) {
		T total;
		WarpExclusiveScan<WARP_SZ>::Add(value, total);
		if (LaneID() == WARP_SZ - 1)
			*pointer = total;
	}

	template<int WARP_SZ>
	template<typename T>
	__device__ __forceinline__
		T WarpExclusiveScan<WARP_SZ>::AddAtom(T& value, T* pointer) {
		T total, prev;
		typedef cub::WarpScan<int> WarpScan;
     
        // Allocate WarpScan shared memory for 4 warps
        __shared__ typename WarpScan::TempStorage temp_storage[16];
     
          // Obtain one input item per thread
     
          // Compute inclusive warp-wide prefix sums
		//total = value;
		// total = value;
		// value = value + 1;
        int warp_id = threadIdx.x / WARP_SZ;
		int lane_id = threadIdx.x % WARP_SZ;
		// if (blockIdx.x == 0 && threadIdx.x < 128)
		// {
		// 	printf("warp scan VW_id = %d, lane_id = %d,threadIdx.x = %d,%d,%d,%d\n",warp_id, lane_id,threadIdx.x, value, *pointer, total);
		// }
		//int agg;
        WarpScan(temp_storage[warp_id]).ExclusiveSum(value, value, total);
		//value = value - lane_id;
		// if (blockIdx.x == 0 && threadIdx.x < 128)
		// {
		// 	printf("warp scan VW_SZ = %d, blockIdx.x = %d,threadIdx.x = %d,%d,%d,%d\n",WARP_SZ, blockIdx.x,threadIdx.x, value, *pointer, total);
		// }
		// if (lane_id == WARP_SZ - 1 && total > 0 )
		// {
		// 	printf("warp scan VW_id = %d, lane_id = %d,threadIdx.x = %d,%d,%d,%d\n",warp_id, lane_id,threadIdx.x, value, *pointer, total);
		// }
		if (lane_id == WARP_SZ - 1)
		{
			//total = total + value;
			// if(total > 0)
			// 	printf("threadIdx = %d, total = %d",threadIdx.x,total);
			prev = atomicAdd(pointer, total);

			/*if (blockIdx.x < 1)
				printf("atomic add blockIdx.x=%d,threadIdx.x = %d,%d,%d,%d,%d\n", blockIdx.x, threadIdx.x, value, *pointer, total);*/
		}
		return __shfl_sync(0xFFFFFFFF,prev,WARP_SZ - 1,WARP_SZ);
     

		//WarpExclusiveScan<WARP_SZ>::Add(value, total);
		// if (blockIdx.x == 0 && threadIdx.x < 32)
		// {
		// 	printf("warp scan blockIdx.x = %d,threadIdx.x = %d,%d,%d,%d\n", blockIdx.x,threadIdx.x, value, *pointer, total);
		// }
		//if (LaneID() == WARP_SZ-1)
		//chl
		//unsigned mask = __ballot_sync(0xFFFFFFFF, total>0);
		/*if(blockIdx.x == 0 && threadIdx.x <8)
			printf("mask:%x,%d,%d,%d", mask, 31 - __clz(mask), WARP_SZ,LaneID());*/
		// if (LaneID() == 31 - __clz(mask))
		// {
		// 	if(total > 0)
		// 		printf("threadIdx = %d, total = %d",threadIdx.x,total);
		// 	prev = atomicAdd(pointer, total);
		// 	/*if (blockIdx.x < 1)
		// 		printf("atomic add blockIdx.x=%d,threadIdx.x = %d,%d,%d,%d,%d\n", blockIdx.x, threadIdx.x, value, *pointer, total);*/
		// }
		// return __shfl(prev, 31 - __clz(mask));
	}

} //@primitives
