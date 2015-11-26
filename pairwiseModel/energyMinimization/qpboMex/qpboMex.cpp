

#include "QPBO.h"
#include "mex.h"

#include <limits>
#include <cmath>

#define INFTY INT_MAX

double round(double a);
int isInteger(double a);

#define MATLAB_ASSERT(expr,msg) if (!(expr)) { mexErrMsgTxt(msg);}

#if !defined(MX_API_VER) || MX_API_VER < 0x07030000
typedef int mwSize;
typedef int mwIndex;
#endif

typedef QPBO<double> GraphType; 

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	MATLAB_ASSERT( nrhs == 2, "qpboMex: Wrong number of input parameters: expected 2");
    MATLAB_ASSERT( nlhs <= 2, "qpboMex: Too many output arguments: expected 2 or less");
	
	//Fix input parameter order:
	const mxArray *uInPtr = (nrhs >= 1) ? prhs[0] : NULL; //unary
	const mxArray *pInPtr = (nrhs >= 2) ? prhs[1] : NULL; //pairwise
	
	//Fix output parameter order:
	mxArray **cOutPtr = (nlhs >= 1) ? &plhs[0] : NULL; //LB
	mxArray **lOutPtr = (nlhs >= 2) ? &plhs[1] : NULL; //labels

	 //node number
	mwSize numNodes;
    
	// get unary potentials
	MATLAB_ASSERT(mxGetNumberOfDimensions(uInPtr) == 2, "qpboMex: The unary paramater is not 2-dimensional");
	MATLAB_ASSERT(mxGetClassID(uInPtr) == mxDOUBLE_CLASS, "qpboMex: Unary potentials are of wrong type");
	MATLAB_ASSERT(mxGetPi(uInPtr) == NULL, "qpboMex: Unary potentials should not be complex");
	
	numNodes = mxGetM(uInPtr);
	
	MATLAB_ASSERT(numNodes >= 1, "qpboMex: The number of nodes is not positive");
	MATLAB_ASSERT(mxGetN(uInPtr) == 2, "qpboMex: The edge paramater is not of size #nodes x 2");
	
	double* termW = (double*)mxGetData(uInPtr);

	//get pairwise potentials
	MATLAB_ASSERT(mxGetNumberOfDimensions(pInPtr) == 2, "qpboMex: The edge paramater is not 2-dimensional");
	
	mwSize numEdges = mxGetM(pInPtr);

	MATLAB_ASSERT( mxGetN(pInPtr) == 6, "qpboMex: The edge paramater is not of size #edges x 6");
	MATLAB_ASSERT(mxGetClassID(pInPtr) == mxDOUBLE_CLASS, "qpboMex: Pairwise potentials are of wrong type");

	double* edges = (double*)mxGetData(pInPtr);
	for(mwSize i = 0; i < numEdges; i++)
	{
		MATLAB_ASSERT(1 <= round(edges[i]) && round(edges[i]) <= numNodes, "qpboMex: error in pairwise terms array");
		MATLAB_ASSERT(isInteger(edges[i]), "qpboMex: error in pairwise terms array");
		MATLAB_ASSERT(1 <= round(edges[i + numEdges]) && round(edges[i + numEdges]) <= numNodes, "qpboMex: error in pairwise terms array");
		MATLAB_ASSERT(isInteger(edges[i + numEdges]), "qpboMex: error in pairwise terms array");
	}



	// start computing
	if (nlhs == 0){
		return;
	}

	//prepare graph
	GraphType *g = new GraphType(numNodes, numEdges);
	
	//add unary potentials
	g -> AddNode(numNodes);
	for(mwSize i = 0; i < numNodes; i++)
	{
		g -> AddUnaryTerm((GraphType::NodeId) i, termW[i], termW[numNodes + i]); 
	}
	
	//add pairwise terms
	for(mwSize i = 0; i < numEdges; i++)
		if(edges[i] < 1 || edges[i] > numNodes || edges[numEdges + i] < 1 || edges[numEdges + i] > numNodes || edges[i] == edges[numEdges + i] || !isInteger(edges[i]) || !isInteger(edges[numEdges + i])){
			mexWarnMsgIdAndTxt("qpboMex:pairwisePotentials", "Some edge has invalid vertex numbers and therefore it is ignored");
		}
		else
		{
			g -> AddPairwiseTerm((GraphType::NodeId) (edges[i] - 1), (GraphType::NodeId) (edges[numEdges + i] - 1), edges[2 * numEdges + i], edges[3 * numEdges + i], edges[4 * numEdges + i], edges[5 * numEdges + i]);
		}

	//Merge edges
	g -> MergeParallelEdges();

	//Solve
	g -> Solve();
	g -> ComputeWeakPersistencies();

	//output lower bound value
	if (cOutPtr != NULL){
		*cOutPtr = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
		*(double*)mxGetData(*cOutPtr) = 0.5 * (g -> ComputeTwiceLowerBound());
	}

	//output labeling
	if (lOutPtr != NULL){
		*lOutPtr = mxCreateNumericMatrix(numNodes, 1, mxDOUBLE_CLASS, mxREAL);
		double* segment = (double*)mxGetData(*lOutPtr);
		for(mwSize i = 0; i < numNodes; i++)
			segment[i] = g -> GetLabel(i);
	}
    
    delete g;
}


double round(double a)
{
	return (mwSize)floor(a + 0.5);
}

int isInteger(double a)
{
	return (a - round(a) < 1e-6);
}
