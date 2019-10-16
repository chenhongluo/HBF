#include "../include/GraphPreDeal.hpp"
#include <../config.cuh>

namespace graph {
    GraphPreDeal::GraphPreDeal(const GraphWeight& gweight):
    gw(gweight),
    InNodesVec(gweight.V+1),
	InEdgesVec(gweight.E),
	OutNodesVec(gweight.V+1),
	OutEdgesVec(gweight.E),
	Orders(gweight.V,ALL_LEVELS-1),
	Candidates(gweight.V),
	Marks(gweight.V)
    {
        V=gweight.V;
        E=gweight.E;
        std::copy(gweight.InNodes,gweight.InNodes+(V+1),&InNodesVec[0]);
        std::copy(gweight.OutNodes,gweight.OutNodes+(V+1),&OutNodesVec[0]);
        // for(int i=0;i<100;i++)
        // {
        //     printf("%d,%d",gweight.InNodes[i],InNodesVec[i]);
        // }
        for(int i=0;i<E;i++)
        {
            int2 temp = make_int2(gweight.InEdges[i],gweight.InWeights[i]);
            InEdgesVec[i] = temp;
            temp = make_int2(gweight.OutEdges[i],gweight.Weights[i]);
            OutEdgesVec[i] = temp;
        }
        for(int i =0;i<gweight.V;i++)
        {
            Candidates[i] = i;
        }
    }
    GraphPreDeal::~GraphPreDeal(){}
}