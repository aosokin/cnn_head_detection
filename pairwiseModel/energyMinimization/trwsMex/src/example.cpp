#include <stdio.h>
#include "MRFEnergy.h"

// Example: minimizing an energy function with Potts terms.
// See type*.h files for other types of terms.

void testPotts()
{
	MRFEnergy<TypePotts>* mrf;
	MRFEnergy<TypePotts>::NodeId* nodes;
	MRFEnergy<TypePotts>::Options options;
	TypePotts::REAL energy, lowerBound;

	const int nodeNum = 2; // number of nodes
	const int K = 3; // number of labels
	TypePotts::REAL D[K];
	int x, y;

	mrf = new MRFEnergy<TypePotts>(TypePotts::GlobalSize(K));
	nodes = new MRFEnergy<TypePotts>::NodeId[nodeNum];

	// construct energy
	D[0] = 0; D[1] = 1; D[2] = 2;
	nodes[0] = mrf->AddNode(TypePotts::LocalSize(), TypePotts::NodeData(D));
	D[0] = 3; D[1] = 4; D[2] = 5;
	nodes[1] = mrf->AddNode(TypePotts::LocalSize(), TypePotts::NodeData(D));
	mrf->AddEdge(nodes[0], nodes[1], TypePotts::EdgeData(6));

	// Function below is optional - it may help if, for example, nodes are added in a random order
	// mrf->SetAutomaticOrdering();

	/////////////////////// TRW-S algorithm //////////////////////
	options.m_iterMax = 30; // maximum number of iterations
	mrf->Minimize_TRW_S(options, lowerBound, energy);

	// read solution
	x = mrf->GetSolution(nodes[0]);
	y = mrf->GetSolution(nodes[1]);

	printf("Solution: %d %d\n", x, y);

	//////////////////////// BP algorithm ////////////////////////
	mrf->ZeroMessages(); // in general not necessary - it may be faster to start 
	                     // with messages computed in previous iterations.
	                     // NOTE: in most cases, immediately after creating the energy
	                     // all messages are zero. EXCEPTION: typeBinary and typeBinaryFast.
	                     // So calling ZeroMessages for these types will NOT transform
	                     // the energy to the original state. 

	options.m_iterMax = 30; // maximum number of iterations
	mrf->Minimize_BP(options, energy);

	// read solution
	x = mrf->GetSolution(nodes[0]);
	y = mrf->GetSolution(nodes[1]);

	printf("Solution: %d %d\n", x, y);

	// done
	delete nodes;
	delete mrf;
}

int main()
{
	testPotts();
	return 0;
}
