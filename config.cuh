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
#include <fstream>
using std::vector;
using std::ofstream;

const unsigned			GRIDDIM = 68*4; 
const unsigned      	BLOCKDIM = 256;
const int             N_OF_TESTS = 1;
const bool CHECK_TRAVERSED_EDGES = false;
const bool          CHECK_RESULT = false;
const int            CUDA_DEBUG = 0;
const int			 DEBUG_LEVEL = 3;
const vector<int> TEST_NODES = 
{
	//USA_CAL_ROAD
	//22216,13227,28177,11679,15587,5823,24036,1846,22574,25324,23666,10894,14436,28113,6765,12511,20460,26121,14998,26166,23982,26028,10800,4122,10534,3608,23989,1904,17590,4337,17865,10652,19009,19046,4204,25833,17909,20662,2675,25643,17256,7411,13625,7587,12519,12551,25746,18722,3339,24308,30976,369,4748,5472,2602,5361,22106,21840,5513,18815,3050,13620,1381,6042,25550,5412,9216,8392,14473,26729,20241,25909,32161,14399,32627,31902,9462,17369,31007,22125,28241,24998,14168,17702,21824,1491,26917,22785,4386,4625,4612,19929,23422,21144,8925,4164,28825,19563,7989,7684,32327,16606,8868,17822,29953,2646,23177,2376,28472,13905,18428,10504,8937,166,24068,6171,4466,5133,20490,24980,27044,13281,9358,16453,20505,13093,24601,28130,455,23750,23667,28004,1238,12575,8213,21676,23471,7362,18913,22889,28246,3190,25873,9531,7673,636,14702,13593,14025,4375,27586,18416,13320,24139,655,8332,2659,5348,8961,7868,27779,26298,6690,29617,15839,1076,1613,30120,10196,11531,13541,22217,1271,15952,2118,17230,5802,25009,31228,10983,14573,16888,3380,14685,2167,26836,17775,17014,12038,10716,30078,9498,18931,16081,31402,27059,7945,31011,14387,8154
	//flickr
	25604,5173,11041,29730,14405,24540,22729,31705,4688,5223,12381,23999,23036,32652,2486,23956,8726,3759,27454,15471,27246,14639,19040,18269,1000,9267,12349,14667,31440,13223,5793,29734,31598,11393,4405,21824,25818,19529,5878,10486,32274,31192,32741,400,330,29536,2816,30056,29059,7596,114,2127,28255,22697,17415,17452,31284,32038,27618,17638,26207,26584,12146,15236,23003,13729,17058,11411,9746,5393,29008,28104,24050,20782,3841,9959,25412,31088,3795,23011,24121,19216,22451,13895,21318,16259,10447,15316,19250,25665,8782,28780,16750,24522,17983,11567,19620,18411,1488,18147,13714,11581,11969,5529,28697,32752,19736,8077,10897,12742,21781,27471,26464,14690,10600,15648,21707,23235,20898,17311,22284,22796,29930,30629,30884,30086,27889,13165,13554,24551,1320,17656,20741,31701,14896,32687,13999,8102,3473,4427,1498,12394,16816,7666,20511,16171,4603,26530,12949,2931,15207,1235,16871,3855,26908,27224,31914,30483,22894,19591,14317,11806,20240,7065,14865,23519,8836,8020,30683,32216,2006,30183,10183,32098,23690,31253,8095,26354,4563,21653,12365,9882,1798,15545,29678,28115,26701,11343,16935,9849,26016,2757,16060,5911,22001,5008,13918,7269,6823,31693
};
// const vector<int>& TESTNODES = TEST_NODES_FOR_CAL_ROAD;
//------------------------------------------------------------------------------
#define LOGMARKS false
#define LOGFRONTIER false
#define LOGNAME "LOG3"

#if LOGFRONTIER|LOGMARKS
	extern ofstream LOGIN;
#endif
#define SHAREDLIMIT 4
#define SHAREDLIMIT2 6

#define REGLIMIT 32

#define MAXSUPLEVEL 100
const float alpha = 0.3;
const float belta = 1.0;

const int ALL_LEVELS = 0;
// const int selectNodesStrategy = 1;
const int selectNodesStrategy = 0;
const vector<int> selectNodesNum = {
	2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
};

#define NODESPLIT false
const int EdgesOneSplit = 4096;
#define SPLITBITS 2

//kernel para
const int KERNELNODES = 68*64;

#define DELTASTEPPING true
const int MAXNODESALLOW = 68*128;

//not to be modified
const int ThreadsOneSplit = 32; // for display
const int MAXSplits = 1 << SPLITBITS;
const int SplitBits = 32 - SPLITBITS;
#if SPLITBITS
const unsigned SplitMask = ((1 << SplitBits) - 1);
#else
const unsigned SplitMask = 0xFFFFFFFF;
#endif
//------------------------------------------------------

#define UPDOWNREM true

#define EDGEALIGN false //TODO
const int AlignSize = 128; // TODO

#define EDGECOMPRESS false//TODO
const int COMPRESSSIZE = 128;//TODO
// 	1000000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000
// };
// 500000,30000,30000,20000,20000,
// 20000,20000,20000,20000,20000,
// 20000,20000,20000,20000,20000,
// 10000,10000,10000,10000,10000,
// 10000,10000,10000,10000,10000};
// const int ALL_LEVELS = 1;
// const vector<int> selectNodesNum = {0,200000,10000,10000,10000,10000,10000,10000,10000,10000};

// const vector<int> upSearchStrategy = {25};
// const vector<int> downSearchStrategy = {25}; 

// -----------------------------------------------------------------------------

#define          ATOMIC64	true    // optimization

const int          MIN_VW = 4;      // minimum size of virtual warp (range [1, 32])
const int		   DEFAULT_VW = 32;
const int          MAX_VW = 32;      // maximum size of virtual warp (range [1, 32])
const int   ITEM_PER_WARP = 1;      // gridDim = RESIDENT_THREADS * ITEM_PER_WARP

//------------------------------------------------------------------------------

#if ATOMIC64
	#define        hdist_t  int2
	#define            INF  (make_int2(0, INT_MAX))
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
