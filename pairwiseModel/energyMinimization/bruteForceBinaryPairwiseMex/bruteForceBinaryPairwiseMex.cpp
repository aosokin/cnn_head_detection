
#include "mex.h"
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>

#define INFTY INT_MAX

double round(double a);
int isInteger(double a);
int minimizeBinaryPairwiseBruteForce(const int numNodes, const int numEdges, const double* unary, const double* pairwise, 
								double &energy, std::vector<int> &labels );

#define MATLAB_ASSERT(expr,msg) if (!(expr)) { mexErrMsgTxt(msg);}

#if !defined(MX_API_VER) || MX_API_VER < 0x07030000
typedef int mwSize;
typedef int mwIndex;
#endif

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	MATLAB_ASSERT( nrhs == 2, "bruteForceBinaryPairwiseMex: Wrong number of input parameters: expected 2");
    MATLAB_ASSERT( nlhs <= 2, "bruteForceBinaryPairwiseMex: Too many output arguments: expected 2 or less");
	
	//Fix input parameter order:
	const mxArray *uInPtr = (nrhs >= 1) ? prhs[0] : NULL; //unary
	const mxArray *pInPtr = (nrhs >= 2) ? prhs[1] : NULL; //pairwise
	
	//Fix output parameter order:
	mxArray **eOutPtr = (nlhs >= 1) ? &plhs[0] : NULL; //energy
	mxArray **lOutPtr = (nlhs >= 2) ? &plhs[1] : NULL; //labels
	
	 //node number
	mwSize numNodes;
    
	// get unary potentials
	MATLAB_ASSERT(mxGetNumberOfDimensions(uInPtr) == 2, "bruteForceBinaryPairwiseMex: The unary paramater is not 2-dimensional");
	MATLAB_ASSERT(mxGetClassID(uInPtr) == mxDOUBLE_CLASS, "bruteForceBinaryPairwiseMex: Unary potentials are of wrong type");
	MATLAB_ASSERT(mxGetPi(uInPtr) == NULL, "bruteForceBinaryPairwiseMex: Unary potentials should not be complex");
	
	numNodes = mxGetM(uInPtr);
	
	MATLAB_ASSERT(numNodes >= 1, "bruteForceBinaryPairwiseMex: The number of nodes is not positive");
	MATLAB_ASSERT(mxGetN(uInPtr) == 2, "bruteForceBinaryPairwiseMex: The edge paramater is not of size #nodes x 2");
	
	double* termW = (double*)mxGetData(uInPtr);

	//get pairwise potentials
	MATLAB_ASSERT(mxGetNumberOfDimensions(pInPtr) == 2, "bruteForceBinaryPairwiseMex: The edge paramater is not 2-dimensional");
	
	mwSize numEdges = mxGetM(pInPtr);

	MATLAB_ASSERT( mxGetN(pInPtr) == 6, "bruteForceBinaryPairwiseMex: The edge paramater is not of size #edges x 6");
	MATLAB_ASSERT(mxGetClassID(pInPtr) == mxDOUBLE_CLASS, "bruteForceBinaryPairwiseMex: Pairwise potentials are of wrong type");

	double* edges = (double*)mxGetData(pInPtr);
	for(mwSize iEdge = 0; iEdge < numEdges; ++iEdge)
	{
		MATLAB_ASSERT(1 <= round(edges[iEdge]) && round(edges[iEdge]) <= numNodes, "bruteForceBinaryPairwiseMex: node index out of bounds, should be in 1,...,numNodes");
		MATLAB_ASSERT(isInteger(edges[iEdge]), "bruteForceBinaryPairwiseMex: non-integer node index");
		MATLAB_ASSERT(1 <= round(edges[iEdge + numEdges]) && round(edges[iEdge + numEdges]) <= numNodes, "bruteForceBinaryPairwiseMex: node index out of bounds, should be in 1,...,numNodes");
		MATLAB_ASSERT(isInteger(edges[iEdge + numEdges]), "bruteForceBinaryPairwiseMex: non-integer node index");
	}

	// start computing
	if (nlhs == 0){
		return;
	}

	const int numLabels = 2;

	double energyValue = DBL_MAX;
	std::vector<int> labels(numNodes, 0);
	minimizeBinaryPairwiseBruteForce(numNodes, numEdges, termW, edges, energyValue, labels);

	//output the energy
	if (eOutPtr != NULL){
		*eOutPtr = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
		*(double*)mxGetData(*eOutPtr) = energyValue;
	}

	//output labeling
	if (lOutPtr != NULL){
		*lOutPtr = mxCreateNumericMatrix(numNodes, 1, mxDOUBLE_CLASS, mxREAL);
		double* segment = (double*)mxGetData(*lOutPtr);
		for(mwSize iNode = 0; iNode < numNodes; ++iNode)
			segment[iNode] = labels[iNode] + 1;
	}
}


struct EdgeBinary{
	int curNode;
	int awayNode;
	double values[4];
};


int minimizeBinaryPairwiseBruteForce_tryLabel(const int numNodes, const double* unary, const std::vector< std::vector<EdgeBinary> > &edges, const int iNode, const double partialEnergy,
								std::vector<int> &labels,  std::vector<int> &bestLabels, double &bestEnergy )
{
	if (iNode == numNodes) {
		if (partialEnergy < bestEnergy) {
			bestEnergy = partialEnergy;
			for(int i = 0; i < numNodes; ++i) {
				bestLabels[i] = labels[i];
			}
		}
		return 0;
	}

	double extraEnergy;
	
	// try different labels
	for(int iLabel = 0; iLabel <= 1; ++iLabel)
	{
		labels[iNode] = iLabel;
		extraEnergy = unary[ iNode + iLabel * numNodes ];
		for(int iEdge = 0; (iEdge < edges[iNode].size()) && (edges[iNode][iEdge].awayNode < iNode); ++iEdge) {
			extraEnergy += edges[iNode][iEdge].values[ 2 * iLabel + labels[ edges[iNode][iEdge].awayNode ] ];
		}

		minimizeBinaryPairwiseBruteForce_tryLabel(numNodes, unary, edges, iNode + 1, partialEnergy + extraEnergy,
								labels,  bestLabels, bestEnergy );
	}
	return 0;
}


bool sortEdgesCriterion(const EdgeBinary& a, const EdgeBinary& b) 
{
    return a.awayNode < b.awayNode;
}

int minimizeBinaryPairwiseBruteForce(const int numNodes, const int numEdges, const double* unary, const double* pairwise, 
								double &energy, std::vector<int> &labels )
{
	labels.resize( numNodes );
	energy = DBL_MAX;

	// convert edges to a convinient data structure
	std::vector< std::vector<EdgeBinary> > edges( numNodes, std::vector<EdgeBinary>(0) );
	for(int iEdge = 0; iEdge < numEdges; ++iEdge) {
		int iNode = (int)round(pairwise[iEdge]) - 1;
		int jNode = (int)round(pairwise[iEdge + numEdges]) - 1;
		
		// add the forward edge
		EdgeBinary curEdgeForward;
		curEdgeForward.curNode = iNode;
		curEdgeForward.awayNode = jNode;
		curEdgeForward.values[0] = pairwise[2 * numEdges + iEdge];
		curEdgeForward.values[1] = pairwise[3 * numEdges + iEdge];
		curEdgeForward.values[2] = pairwise[4 * numEdges + iEdge];
		curEdgeForward.values[3] = pairwise[5 * numEdges + iEdge];
		edges[iNode].push_back(curEdgeForward);
	
		// add the backward edge
		EdgeBinary curEdgeBackward;
		curEdgeBackward.curNode = jNode;
		curEdgeBackward.awayNode = iNode;
		curEdgeBackward.values[0] = pairwise[2 * numEdges + iEdge];
		curEdgeBackward.values[1] = pairwise[4 * numEdges + iEdge]; // note that here the order of potentials values is changed
		curEdgeBackward.values[2] = pairwise[3 * numEdges + iEdge];
		curEdgeBackward.values[3] = pairwise[5 * numEdges + iEdge];
		edges[jNode].push_back( curEdgeBackward );
	}

	// sort all the edges with the increasing order of the away edges
	for(int iNode = 0; iNode < numNodes; ++iNode) {
		std::sort(edges[iNode].begin(), edges[iNode].end(), sortEdgesCriterion);		
	}

	std::vector<int> curLabels(numNodes, 0);
	minimizeBinaryPairwiseBruteForce_tryLabel(numNodes, unary, edges, 0, 0.0,
								curLabels,  labels, energy );

	return 0;
}


double round(double a)
{
	return (mwSize)floor(a + 0.5);
}

int isInteger(double a)
{
	return (abs(a - round(a)) < 1e-6);
}
