
#include "mex.h"
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>

#define INFTY INT_MAX
#define NUM_LABELS 2

double round(double a);
int isInteger(double a);
int computeMinMarginalsBinaryPairwiseBruteForce(const int numNodes, const int numEdges, const double* unary, const double* pairwise, 
								std::vector< std::vector<double> > &minMarginalValues, 
								std::vector< std::vector< std::vector<int> > > &minMarginalArgs );

#define MATLAB_ASSERT(expr,msg) if (!(expr)) { mexErrMsgTxt(msg);}

#if !defined(MX_API_VER) || MX_API_VER < 0x07030000
typedef int mwSize;
typedef int mwIndex;
#endif

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	MATLAB_ASSERT( nrhs == 2, "computeMinMarginalsBinaryMex: Wrong number of input parameters: expected 2");
    MATLAB_ASSERT( nlhs <= 2, "computeMinMarginalsBinaryMex: Too many output arguments: expected 2 or less");
	
	//Fix input parameter order:
	const mxArray *uInPtr = (nrhs >= 1) ? prhs[0] : NULL; //unary
	const mxArray *pInPtr = (nrhs >= 2) ? prhs[1] : NULL; //pairwise
	
	//Fix output parameter order:
	mxArray **valuesOutPtr = (nlhs >= 1) ? &plhs[0] : NULL; // min-marginal values
	mxArray **argsOutPtr = (nlhs >= 2) ? &plhs[1] : NULL; // min-marginal args
	
	 //node number
	mwSize numNodes;
    
	// get unary potentials
	MATLAB_ASSERT(mxGetNumberOfDimensions(uInPtr) == 2, "computeMinMarginalsBinaryMex: The unary paramater is not 2-dimensional");
	MATLAB_ASSERT(mxGetClassID(uInPtr) == mxDOUBLE_CLASS, "computeMinMarginalsBinaryMex: Unary potentials are of wrong type");
	MATLAB_ASSERT(mxGetPi(uInPtr) == NULL, "computeMinMarginalsBinaryMex: Unary potentials should not be complex");
	
	numNodes = mxGetM(uInPtr);
	
	MATLAB_ASSERT(numNodes >= 1, "computeMinMarginalsBinaryMex: The number of nodes is not positive");
	MATLAB_ASSERT(mxGetN(uInPtr) == NUM_LABELS, "computeMinMarginalsBinaryMex: The edge paramater is not of size #nodes x 2");
	
	double* termW = (double*)mxGetData(uInPtr);

	//get pairwise potentials
	MATLAB_ASSERT(mxGetNumberOfDimensions(pInPtr) == 2, "computeMinMarginalsBinaryMex: The edge paramater is not 2-dimensional");
	
	mwSize numEdges = mxGetM(pInPtr);

	MATLAB_ASSERT( mxGetN(pInPtr) == 6, "computeMinMarginalsBinaryMex: The edge paramater is not of size #edges x 6");
	MATLAB_ASSERT(mxGetClassID(pInPtr) == mxDOUBLE_CLASS, "computeMinMarginalsBinaryMex: Pairwise potentials are of wrong type");

	double* edges = (double*)mxGetData(pInPtr);
	for(mwSize iEdge = 0; iEdge < numEdges; ++iEdge)
	{
		MATLAB_ASSERT(1 <= round(edges[iEdge]) && round(edges[iEdge]) <= numNodes, "computeMinMarginalsBinaryMex: node index out of bounds, should be in 1,...,numNodes");
		MATLAB_ASSERT(isInteger(edges[iEdge]), "computeMinMarginalsBinaryMex: non-integer node index");
		MATLAB_ASSERT(1 <= round(edges[iEdge + numEdges]) && round(edges[iEdge + numEdges]) <= numNodes, "computeMinMarginalsBinaryMex: node index out of bounds, should be in 1,...,numNodes");
		MATLAB_ASSERT(isInteger(edges[iEdge + numEdges]), "computeMinMarginalsBinaryMex: non-integer node index");
	}

	// start computing
	if (nlhs == 0){
		return;
	}

	const int numLabels = NUM_LABELS;

	std::vector< std::vector<double> > minMarginalValues(numNodes, std::vector<double> ( numLabels, DBL_MAX ));
	std::vector< std::vector< std::vector<int> > > minMarginalArgs(numNodes, std::vector< std::vector<int> > ( numLabels, std::vector<int>(numNodes, 0) ) );

	computeMinMarginalsBinaryPairwiseBruteForce(numNodes, numEdges, termW, edges, minMarginalValues, minMarginalArgs );

	// output the values of the min-marginals
	if (valuesOutPtr != NULL){
		*valuesOutPtr = mxCreateNumericMatrix(numNodes, numLabels, mxDOUBLE_CLASS, mxREAL);
		double* values = (double*)mxGetData(*valuesOutPtr);
		for(mwSize iNode = 0; iNode < numNodes; ++iNode)
			for(mwSize iLabel = 0; iLabel < numLabels; ++iLabel) {
				values[iNode + iLabel * numNodes] = minMarginalValues[iNode][iLabel];
			}
	}

	// output the arguments of the min-marginals
	if (argsOutPtr != NULL){
		mwSize numDims = 3;
		mwSize dims[3] = { numNodes, numLabels, numNodes };
		*argsOutPtr = mxCreateNumericArray(numDims, dims, mxDOUBLE_CLASS, mxREAL);

		double* args = (double*)mxGetData(*argsOutPtr);
		for(mwSize iNode = 0; iNode < numNodes; ++iNode)
			for(mwSize iLabel = 0; iLabel < numLabels; ++iLabel)
				for(mwSize jNode = 0; jNode < numNodes; ++jNode) {
					args[ iNode + iLabel * numNodes + jNode * numLabels * numNodes ] = minMarginalArgs[iNode][iLabel][jNode];
				}
			
	}
}


struct EdgeBinary{
	int curNode;
	int awayNode;
	double values[4];
};


int computeMinMarginalsBinaryPairwiseBruteForce_tryLabel(const int numNodes, const double* unary, const std::vector< std::vector<EdgeBinary> > &edges, const int iNode, const double partialEnergy,
								std::vector<int> &labels, 
								std::vector< std::vector<double> > &minMarginalValues, 
								std::vector< std::vector< std::vector<int> > > &minMarginalArgs )
{
	if (iNode == numNodes) {
		// check the current labeling: labels
		for(int i = 0; i < numNodes; ++i) {
			if (partialEnergy < minMarginalValues[i][ labels[i] ]) {
				minMarginalValues[i][ labels[i] ] = partialEnergy;
				for(int j = 0; j < numNodes; ++j) {
					minMarginalArgs[i][ labels[i] ][j] = labels[j];
				}
			}
		}
		return 0;
	}

	double extraEnergy;
	
	// try different labels
	for(int iLabel = 0; iLabel < NUM_LABELS; ++iLabel)
	{
		labels[iNode] = iLabel;
		extraEnergy = unary[ iNode + iLabel * numNodes ];
		for(int iEdge = 0; (iEdge < edges[iNode].size()) && (edges[iNode][iEdge].awayNode < iNode); ++iEdge) {
			extraEnergy += edges[iNode][iEdge].values[ NUM_LABELS * iLabel + labels[ edges[iNode][iEdge].awayNode ] ];
		}

		computeMinMarginalsBinaryPairwiseBruteForce_tryLabel(numNodes, unary, edges, iNode + 1, partialEnergy + extraEnergy,
								labels, minMarginalValues, minMarginalArgs );
	}
	return 0;
}


bool sortEdgesCriterion(const EdgeBinary& a, const EdgeBinary& b) 
{
    return a.awayNode < b.awayNode;
}

int computeMinMarginalsBinaryPairwiseBruteForce(const int numNodes, const int numEdges, const double* unary, const double* pairwise, 
								std::vector< std::vector<double> > &minMarginalValues, 
								std::vector< std::vector< std::vector<int> > > &minMarginalArgs )
{
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
	computeMinMarginalsBinaryPairwiseBruteForce_tryLabel(numNodes, unary, edges, 0, 0.0,
								curLabels, minMarginalValues, minMarginalArgs );

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
