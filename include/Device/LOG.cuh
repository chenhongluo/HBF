#include <cub/cub.cuh>
#include <XLib.hpp>
#include <../config.cuh>
namespace Kernels
{

#if LOGFRONTIER
void LogPrintfFrotier(int *intvec, int *len, char *s)
{
    LOGIN << s << " ";
    int temp;
    copyIntFrom(temp, len);
    vector<int> vt(temp);
    copyVecFrom<int>(vt, intvec);
    for (int i = 0; i < vt.size(); i++)
    {
        LOGIN << vt[i] << " ";
    }
    LOGIN << std::endl;
}

void LogPrintfStr(char *s)
{
    LOGIN << s << " " << std::endl;
}

#endif

} // namespace Kernels