#include <cub/cub.cuh>
#include <XLib.hpp>
#include <../config.cuh>
#include "LOG.cuh"

namespace Kernels
{

    template <int VW_SIZE>
	__global__ void GNRSearchUp(
        const int* __restrict__ devUpOrderTrans,
        const int* __restrict__ devDownOrderTrans,
        int* __restrict__  devNodes,
		int2* __restrict__  devEdges,
		hdist_t* __restrict__ devDistances,
		int* __restrict__  devF1,
		int* __restrict__  devF2,
        const int* pdevF1Size,
        int* pdevF2Size,
        int** upBuckets,
        int** upBucketSize,
        int** downBuckets,
        int** downBucketSize,
		const int suplevel,
        const int sublevel)
    {
        //printf("dfsafdsa");
        int Queue[REG_LIMIT];
        int founds=0;
        thread_block g = this_thread_block();
        thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
        const int devF1Size = cub::ThreadLoad<cub::LOAD_CG>(pdevF1Size);
        const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / tile.size();
        const int Stride = gridDim.x * (BLOCKDIM / tile.size());
        tile.sync();        
        // devPrintfInt(1,devF1Size,"devF1Size");
        for (int i = VirtualID; i < devF1Size; i+=Stride)
        {
            int index = cub::ThreadLoad<cub::LOAD_CS> (devF1+i);tile.sync();
            //devPrintfInt(32,index,"index");
            hdist_t nodeWeight = cub::ThreadLoad<cub::LOAD_CG>(devDistances + index);tile.sync();
            edge_t start = cub::ThreadLoad<cub::LOAD_CG>(devNodes+index);tile.sync();
            edge_t end = cub::ThreadLoad<cub::LOAD_CG>(devNodes+index+1);tile.sync();
            // devPrintfInt(32,start,"start");
            // devPrintfInt(32,end,"end");
            // devPrintfInt(32,index,"index");
            // devPrintfInt2(32,nodeWeight,"nodeWeight");
            for (int k = start+tile.thread_rank();k<end+tile.thread_rank();k+=tile.size())
            {
                int2 dest;
                if(k<end)
                {
                //    dest = cub::ThreadLoad<cub::LOAD_CG>(devEdges + k);
                    dest = devEdges[k];
                }
                // devPrintfInt2(32,dest,"dest");
                tile.sync();
                if(k<end)
                {
                    if (nodeWeight.y != INT_MAX)
                    {
#if ATOMIC64
                        int level = suplevel<<16 | sublevel;
                        int newWeight = nodeWeight.y + dest.y;
                        //TODO--chl  level--newWeight?
                        // devPrintfInt(1,suplevel,"suplevel");
                        // devPrintfInt(1,sublevel,"sublevel");
                        // devPrintfX(1,level,"level");
                        int2 toWrite = make_int2(level, newWeight);
                        unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
                                                        reinterpret_cast<unsigned long long &>(toWrite));
                        tile.sync();
                        int2 toTest = reinterpret_cast<int2 &>(aa);
                        int testSuplevel = toTest.x>>16;
                        int testSublevel = toTest.x&0x0000FFFF;
                        // devPrintfInt(32,testSuplevel,"testSuplevel");
                        // devPrintfInt(32,testSublevel,"testSublevel");
                        // devPrintfX(32,toTest.x,"toTest.x");
                        //printf("%0x,id = %d\n",toTest.x,blockIdx.x * BLOCKDIM + threadIdx.x);
                        
                        // devPrintfInt2(32,toTest,"toTest");
                        if (toTest.y > newWeight)
                        {
                            int flagForBucket = (toTest.y==INT_MAX)?1:0;
                            // devPrintfInt(32,flagForBucket,"flagForBucket");
#else
                        int oldWeight = atomicMin(&devDistances[dest.x], newWeight);
                        tile.sync();
                        if (oldWeight > newWeight)
                        {
                            int flagForBucket = (oldWeight==INT_MAX)?1:0;
#endif
                            int dsl = cub::ThreadLoad<cub::LOAD_CG>(devUpOrderTrans+dest.x);
                            // devPrintfInt(32,dsl,"dsl");
                            if(dsl > suplevel && flagForBucket)
                            {
                                int val=1;
                                int *bucketMem = cub::ThreadLoad<cub::LOAD_CA> (upBuckets+dsl);
                                int *bucketSizeMem = cub::ThreadLoad<cub::LOAD_CA> (upBucketSize+dsl);
                                //int* bucketSize=cub::ThreadLoad<cub::LOAD_CA> (bucketSizes+dsl)
                                int bias = atomicAdd(bucketSizeMem,val);
                                cub::ThreadStore<cub::STORE_CG>(bucketMem+bias,dest.x);
                            }
                            else if(dsl == suplevel && (testSuplevel!= suplevel || testSublevel != sublevel))
                            {
                                Queue[founds++] = dest.x;
                            }
                            else if(dsl< suplevel)
                            {
                                printf("Error Up Graph");
                            }
                            
                            tile.sync();
                            dsl = cub::ThreadLoad<cub::LOAD_CG>(devDownOrderTrans+dest.x);
                            tile.sync();
                            // devPrintfInt(32,dsl,"dsl");
                            if(flagForBucket)
                            {
                                int val=1;
                                int *bucketMem = cub::ThreadLoad<cub::LOAD_CA> (downBuckets+dsl);
                                // devPrintfInt(32,(int)bucketMem,"bucketMem");
                                int *bucketSizeMem = cub::ThreadLoad<cub::LOAD_CA> (downBucketSize+dsl);
                                // devPrintfInt(32,(int)bucketSizeMem,"bucketSizeMem");
                                //int* bucketSize=cub::ThreadLoad<cub::LOAD_CA> (bucketSizes+dsl)
                                int bias = atomicAdd(bucketSizeMem,val);
                                // devPrintfInt(32,(int)bias,"bias");
                                cub::ThreadStore<cub::STORE_CG>(bucketMem+bias,dest.x);
                                // devPrintfInt(32,(int)bias,"bias");
                            }
                        }
                        else
                        {
                            tile.sync();
                            tile.sync();
                        }
                    }
                }
                else
                {
                    tile.sync();
                    tile.sync();
                    tile.sync();
                }
                // devPrintfInt(32,founds,"founds");
                tile.sync();
                if(tile.any(founds>=REG_LIMIT))
                {
                    VWWrite<VW_SIZE,int>(tile,pdevF2Size,devF2,founds,Queue);
                    founds = 0;
                }
                tile.sync();
            }
        }
        VWWrite<VW_SIZE,int>(tile,pdevF2Size,devF2,founds,Queue);
        founds = 0;
	}

    template<int VW_SIZE>
	__global__ void GNRSearchDown(
        const int* __restrict__ devUpOrderTrans,
        const int* __restrict__ devDownOrderTrans,
        int* __restrict__  devNodes,
		int2* __restrict__    devEdges,
		hdist_t* __restrict__ devDistances,
		int* __restrict__  devF1,
		int* __restrict__  devF2,
        const int* pdevF1Size,
        int* pdevF2Size,
        int** __restrict__ upBuckets,
        int** __restrict__ upBucketSize,
        int** __restrict__ downBuckets,
        int** __restrict__ downBucketSize,
		const int suplevel,
        const int sublevel) 
    {  
        int Queue[REG_LIMIT];
        int founds=0;
        thread_block g = this_thread_block();
        thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
        const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / tile.size();
        const int Stride = gridDim.x * (BLOCKDIM / tile.size());
        const int devF1Size = cub::ThreadLoad<cub::LOAD_CA>(pdevF1Size);
        for (int i = VirtualID; i < devF1Size; i+=Stride)
        {
            int index = cub::ThreadLoad<cub::LOAD_CS> (devF1+i);
            hdist_t nodeWeight = cub::ThreadLoad<cub::LOAD_CS>(devDistances + index);
            edge_t start = cub::ThreadLoad<cub::LOAD_CS>(devNodes+index);
            edge_t end = cub::ThreadLoad<cub::LOAD_CS>(devNodes+index+1);
            // devPrintfInt(1,index,"index");
            // devPrintfInt(1,start,"start");
            // devPrintfInt(1,end,"end");
            // devPrintfInt2(1,nodeWeight,"nodeWeight");
            for (int k = start+tile.thread_rank();k<end+tile.thread_rank();k+=tile.size())
            {
                int2 dest;
                if(k<end)
                {
                    dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdges + k);
                    //devPrintfInt2(1,dest,"dest");
                }
                tile.sync();
                if(k<end)
                {
                    if (nodeWeight.y != INT_MAX)
                    {
#if ATOMIC64
                        int newWeight = nodeWeight.y + dest.y;
                        
                        //TODO--chl level--newWeight?
                        int level = suplevel<<16 | sublevel;
                        level = level | 0x40000000;
                        // devPrintfInt(32,suplevel,"suplevel");
                        // devPrintfInt(32,sublevel,"sublevel");
                        // devPrintfX(32,level,"level");
                        int2 toWrite = make_int2(level, newWeight);
                        // devPrintfulong(32,reinterpret_cast<unsigned long long &>(toWrite),"towrite");
                        // devPrintfulong(32,reinterpret_cast<unsigned long long &>(devDistances[dest.x]),"tomin");
                        unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
                                                        reinterpret_cast<unsigned long long &>(toWrite));
                        tile.sync();
                        int2 toTest = reinterpret_cast<int2 &>(aa);
                        int isDown = toTest.x>>30;
                        int testSuplevel = (toTest.x & 0x3FFFFFFF)>>16;
                        int testSublevel = toTest.x&0x0000FFFF;
                        // devPrintfInt(1,isDown,"isDown");
                        // devPrintfInt2(1,toTest,"toTest");
                        // devPrintfInt(32,testSuplevel,"testSuplevel");
                        // devPrintfInt(32,testSublevel,"testSublevel");
                        // devPrintfX(32,toTest.x,"toTest.x");
                        if (toTest.y > newWeight)
                        {
                            int flagForBucket = (toTest.y==INT_MAX)?1:0;
#else
                        const int oldWeight = atomicMin(&devDistances[dest.x], newWeight);
                        tile.sync();
                        if (oldWeight > newWeight)
                        {
                            int flagForBucket = (oldWeight==INT_MAX)?1:0;
#endif
                            int dsl = cub::ThreadLoad<cub::LOAD_CG>(devDownOrderTrans+dest.x);
                            if(dsl < suplevel && flagForBucket)
                            {
                                int val=1;
                                int *bucketMem = cub::ThreadLoad<cub::LOAD_CA> (downBuckets+dsl);
                                int *bucketSizeMem = cub::ThreadLoad<cub::LOAD_CA> (downBucketSize+dsl);
                                //int* bucketSize=cub::ThreadLoad<cub::LOAD_CA> (bucketSizes+dsl)
                                int bias = atomicAdd(bucketSizeMem,val);
                                cub::ThreadStore<cub::STORE_CG>(bucketMem+bias,dest.x);
                            }
                            else if(dsl == suplevel && (!isDown || testSuplevel != suplevel || testSublevel != sublevel))
                            {
                                Queue[founds++] = dest.x;
                            }
                            else if(dsl > suplevel)
                            {
                                printf("Error Down Graph");
                            }
                            
                        }
                    }
                    else
                    {
                        printf("Error Down Graph");
                    }
                    
                }
                else
                {
                    tile.sync();
                }
                
                if(tile.any(founds>=REG_LIMIT))
                {
                    VWWrite<VW_SIZE,int>(tile,pdevF2Size,devF2,founds,Queue);
                    founds = 0;
                }
            }
        }
        //devPrintfInt(32,founds,"founds");
        VWWrite<VW_SIZE,int>(tile,pdevF2Size,devF2,founds,Queue);
        founds = 0;
    }

	__global__ void GNRSearchUpForNodeSplit(
        const int* __restrict__ devUpOrderTrans,
        const int* __restrict__ devDownOrderTrans,
        int* __restrict__  devNodes,
		int2* __restrict__  devEdges,
        int* __restrict__ devNodeSplit,
		hdist_t* __restrict__ devDistances,
		int* __restrict__  devF1,
		int* __restrict__  devF2,
        const int* pdevF1Size,
        int* pdevF2Size,
        int** upBuckets,
        int** upBucketSize,
        int** downBuckets,
        int** downBucketSize,
		const int suplevel,
        const int sublevel)
    {
        //printf("dfsafdsa");
        int Queue[REG_LIMIT];
        int founds=0;
        thread_block g = this_thread_block();
        thread_block_tile<32> tile = tiled_partition<32>(g);
        const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / tile.size();
        const int Stride = gridDim.x * (BLOCKDIM / tile.size());
        const int devF1Size = cub::ThreadLoad<cub::LOAD_CA>(pdevF1Size);
        // devPrintfInt(1,devF1Size,"devF1Size");
        for (int i = VirtualID; i < devF1Size; i+=Stride)
        {
            int inf_index = cub::ThreadLoad<cub::LOAD_CS> (devF1+i);
            int splitIndex = inf_index>>(32-LOGSplit);
            int index = inf_index & (0xFFFFFFFF >> LOGSplit);
            int splitAll = cub::ThreadLoad<cub::LOAD_CS>(devNodeSplit+index);
            //devPrintfInt(32,index,"index");
            hdist_t nodeWeight = cub::ThreadLoad<cub::LOAD_CS>(devDistances + index);
            edge_t start = cub::ThreadLoad<cub::LOAD_CS>(devNodes+index);
            edge_t end = cub::ThreadLoad<cub::LOAD_CS>(devNodes+index+1);
            // devPrintfInt(32,start,"start");
            // devPrintfInt(32,end,"end");
            // devPrintfInt(32,index,"index");
            // devPrintfInt2(32,nodeWeight,"nodeWeight");
            for (int k = start+splitIndex * 32+tile.thread_rank();
            k<end+splitIndex * 32 + tile.thread_rank();
            k+=splitAll * 32)
            {
                int2 dest;
                if(k<end)
                {
                   dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdges + k);
                }
                // devPrintfInt2(32,dest,"dest");
                tile.sync();
                if(k<end)
                {
                    if (nodeWeight.y != INT_MAX)
                    {
#if ATOMIC64
                        int level = suplevel<<16 | sublevel;
                        int newWeight = nodeWeight.y + dest.y;
                        //TODO--chl  level--newWeight?
                        // devPrintfInt(1,suplevel,"suplevel");
                        // devPrintfInt(1,sublevel,"sublevel");
                        // devPrintfX(1,level,"level");
                        int2 toWrite = make_int2(level, newWeight);
                        unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
                                                        reinterpret_cast<unsigned long long &>(toWrite));
                        tile.sync();
                        int2 toTest = reinterpret_cast<int2 &>(aa);
                        int testSuplevel = toTest.x>>16;
                        int testSublevel = toTest.x&0x0000FFFF;
                        // devPrintfInt(32,testSuplevel,"testSuplevel");
                        // devPrintfInt(32,testSublevel,"testSublevel");
                        // devPrintfX(32,toTest.x,"toTest.x");
                        //printf("%0x,id = %d\n",toTest.x,blockIdx.x * BLOCKDIM + threadIdx.x);
                        
                        // devPrintfInt2(32,toTest,"toTest");
                        if (toTest.y > newWeight)
                        {
                            int flagForBucket = (toTest.y==INT_MAX)?1:0;
                            // devPrintfInt(32,flagForBucket,"flagForBucket");
#else
                        int oldWeight = atomicMin(&devDistances[dest.x], newWeight);
                        tile.sync();
                        if (oldWeight > newWeight)
                        {
                            int flagForBucket = (oldWeight==INT_MAX)?1:0;
#endif
                            int dsl = cub::ThreadLoad<cub::LOAD_CG>(devUpOrderTrans+dest.x);
                            // devPrintfInt(32,dsl,"dsl");
                            if(dsl > suplevel && flagForBucket)
                            {
                                int val=1;
                                int *bucketMem = cub::ThreadLoad<cub::LOAD_CA> (upBuckets+dsl);
                                int *bucketSizeMem = cub::ThreadLoad<cub::LOAD_CA> (upBucketSize+dsl);
                                //int* bucketSize=cub::ThreadLoad<cub::LOAD_CA> (bucketSizes+dsl)
                                int bias = atomicAdd(bucketSizeMem,val);
                                cub::ThreadStore<cub::STORE_CG>(bucketMem+bias,dest.x);
                            }
                            else if(dsl == suplevel && (testSuplevel!= suplevel || testSublevel != sublevel))
                            {
                                Queue[founds++] = dest.x;
                            }
                            else if(dsl< suplevel)
                            {
                                printf("Error Up Graph");
                            }
                            
                            tile.sync();
                            dsl = cub::ThreadLoad<cub::LOAD_CG>(devDownOrderTrans+dest.x);
                            // devPrintfInt(32,dsl,"dsl");
                            if(flagForBucket)
                            {
                                int val=1;
                                int *bucketMem = cub::ThreadLoad<cub::LOAD_CA> (downBuckets+dsl);
                                // devPrintfInt(32,(int)bucketMem,"bucketMem");
                                int *bucketSizeMem = cub::ThreadLoad<cub::LOAD_CA> (downBucketSize+dsl);
                                // devPrintfInt(32,(int)bucketSizeMem,"bucketSizeMem");
                                //int* bucketSize=cub::ThreadLoad<cub::LOAD_CA> (bucketSizes+dsl)
                                int bias = atomicAdd(bucketSizeMem,val);
                                // devPrintfInt(32,(int)bias,"bias");
                                cub::ThreadStore<cub::STORE_CG>(bucketMem+bias,dest.x);
                                // devPrintfInt(32,(int)bias,"bias");
                            }
                        }
                        else
                        {
                            tile.sync();
                        }
                    }
                }
                else
                {
                    tile.sync();
                    tile.sync();
                }
                // devPrintfInt(32,founds,"founds");
                if(tile.any(founds>=REG_LIMIT))
                {
                    VWWrite<32,int>(tile,pdevF2Size,devF2,founds,Queue);
                    founds = 0;
                }
            }
        }
        VWWrite<32,int>(tile,pdevF2Size,devF2,founds,Queue);
        founds = 0;
	}
}