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
#ifndef _CONFIG_H
#define _CONFIG_H
#include <vector>
using std::vector;

const unsigned			GRIDDIM = 128; 
const unsigned      	BLOCKDIM = 256;
const int             N_OF_TESTS = 100;
const bool CHECK_TRAVERSED_EDGES = false;
const bool          CHECK_RESULT = true;
const int            CUDA_DEBUG = 0;
const int			 DEBUG_LEVEL = 3;
//------------------------------------------------------------------------------

#define MAXSUPLEVEL 100

const int selectNodesStrategy = 1;
const int ALL_LEVELS = 10;
const vector<int> selectNodesNum = {200000,200000,10000,10000,10000,10000,10000,10000,10000,10000};
// const int ALL_LEVELS = 1;
// const vector<int> selectNodesNum = {0,200000,10000,10000,10000,10000,10000,10000,10000,10000};

const vector<int> upSearchStrategy = {10};
const vector<int> downSearchStrategy = {10}; 

// -----------------------------------------------------------------------------

#define          ATOMIC64	true    // optimization

const int          MIN_VW = 4;      // minimum size of virtual warp (range [1, 32])
const int		   DEFAULT_VW = 8;
const int          MAX_VW = 32;      // maximum size of virtual warp (range [1, 32])
const int   ITEM_PER_WARP = 1;      // gridDim = RESIDENT_THREADS * ITEM_PER_WARP

//------------------------------------------------------------------------------

const int REG_LIMIT = 32;            // register size

#if ATOMIC64
	#define        hdist_t  int2
	#define            INF  (make_int2(INT_MAX, INT_MAX))
	#define           ZERO	(make_int2(0, 0))
	#define MAKE_DIST(a,b)	( make_int2((a), (b)) )
#else
	#define	       hdist_t	int
	#define            INF  INT_MAX
	#define           ZERO	0
	#define MAKE_DIST(a,b)	( b )
#endif

// static_assert(!DYNAMIC_PARALLELISM || !GLOBAL_SYNC,
//     "dynamic parallelism and global sync must not be enabled at the same time ");

#endif
