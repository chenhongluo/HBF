#include <cub/cub.cuh>
#include <XLib.hpp>
#include <../config.cuh>
namespace Kernels
{
    void hostPrintfIntVec(int *intvec,int* len,char* s)
    {       
        printf("%s:",s);
        int temp;
        copyIntFrom(len,&temp);
        printf("%d\n",temp);
        // vector<int> vt(temp);
        // copyVecFrom<int>(intvec,vt);
        // for(int i=0;i<vt.size();i++)
        // {
        //     printf("%d ",vt[i]);
        // }
        // printf("\n");
    }

#if LOGFRONTIER
    void LogPrintfFrotier(int *intvec,int* len,char* s)
    {
        LOGIN<<s<<" ";
        int temp;
        copyIntFrom(len,&temp);
        vector<int> vt(temp);
        copyVecFrom<int>(intvec,vt);
        for(int i=0;i<vt.size();i++)
        {
            LOGIN<<vt[i]<<" ";
        }
        LOGIN<<std::endl;
    }

    void LogPrintfStr(char* s)
    {
        LOGIN<<s<<" "<<std::endl;
    }
#endif
    __device__ void devPrintfInt(int k,int n,char* s)
    {
        if(blockIdx.x * BLOCKDIM + threadIdx.x < k)
        {
            printf("%s:%d,id:%d\n",s,n,blockIdx.x * BLOCKDIM + threadIdx.x);
        }
    }

    __device__ void devPrintfX(int k,int n,char* s)
    {
        if(blockIdx.x * BLOCKDIM + threadIdx.x < k)
        {
            printf("%s:%0x,id:%d\n",s,n,blockIdx.x * BLOCKDIM + threadIdx.x);
        }
    }

    __device__ void devPrintfInt2(int k,int2 n,char* s)
    {
        if(blockIdx.x * BLOCKDIM + threadIdx.x < k)
        {
            printf("%s:%d,%d,id:%d\n",s,n.x,n.y,blockIdx.x * BLOCKDIM + threadIdx.x);
        }
    }

    __device__ void devPrintfulong(int k,unsigned long long n,char* s)
    {
        if(blockIdx.x * BLOCKDIM + threadIdx.x < k)
        {
            printf("%s:%lld,%d,id:%d\n",s,n,sizeof(n),blockIdx.x * BLOCKDIM + threadIdx.x);
        }
    }
}