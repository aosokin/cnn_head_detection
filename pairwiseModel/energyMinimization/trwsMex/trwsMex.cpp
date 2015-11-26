#include <cstdlib>
#include <stdio.h>
#include <limits>
#include <time.h>
#include <vector>
using std::vector;

#include "src/MRFEnergy.h"
#include "mex.h"

#define MATLAB_ASSERT(expr,msg) if (!(expr)) {mexErrMsgIdAndTxt( "trwsMex_time:error", msg);}

#if !defined(MX_API_VER) || MX_API_VER < 0x07030000
typedef int mwSize;
typedef int mwIndex;
#endif

int verbosityLevel; 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	MATLAB_ASSERT(nrhs >= 2 , "Not enough input arguments, expected 2 - 4" ); \
	MATLAB_ASSERT(nrhs <= 4, "Too many input arguments, expected 2 - 4"); 	

	MATLAB_ASSERT(nlhs <= 6, "Too many output arguments, expected 1 - 6");
	
	//Fix input parameter order:
	const mxArray *uInPtr = (nrhs > 0) ? prhs[0] : NULL; //unary
	const mxArray *pInPtr = (nrhs > 1) ? prhs[1] : NULL; //pairwise
	const mxArray *mInPtr = (nrhs > 2) ? prhs[2] : NULL; //label matrix
	const mxArray *oInPtr = (nrhs > 3) ? prhs[3] : NULL; //options
	
	//Fix output parameter order:
	mxArray **eOutPtr = (nlhs > 1) ? &plhs[1] : NULL; //energy
	mxArray **sOutPtr = (nlhs > 0) ? &plhs[0] : NULL; //solution
	mxArray **lbOutPtr = (nlhs > 2) ? &plhs[2] : NULL; //lowerbound
	mxArray **timePlotOutPtr = (nlhs > 5) ? &plhs[5] : NULL; //time plot 
	mxArray **lbPlotOutPtr = (nlhs > 3) ? &plhs[3] : NULL; //lowerbound plot 
	mxArray **energyPlotOutPtr = (nlhs > 4) ? &plhs[4] : NULL; //energy plot 

	//prepare default options
	double m_eps = 1e-2; 
	int m_iterMax = 100;
	int m_printIter = 5;     
	int m_printMinIter = 10;
	verbosityLevel = 0; // global variable
	int method = 0;
	
	//get options structure
	if(oInPtr != NULL){
		MATLAB_ASSERT(mxIsStruct(oInPtr), "Expected structure array for constraints");
		MATLAB_ASSERT(mxGetNumberOfElements(oInPtr) == 1, "Wrong structure type for options: wrong number of fields");
		mxArray *curField = NULL;
		if((curField = mxGetField(oInPtr, 0, "method")) != NULL){
			MATLAB_ASSERT(mxGetClassID(curField) == mxCHAR_CLASS, "Wrong structure type for options: expected STRING for field <<method>>");
		
			mwSize buflen = mxGetN(curField)*sizeof(mxChar)+1;
			char *buf = (char*)mxMalloc(buflen);
			if(!mxGetString(curField, buf, buflen)){
				if(!strcmp(buf, "trw-s")) method = 0;
				if(!strcmp(buf, "bp")) method = 1;
			}
			mxFree(buf);
		}
		if((curField = mxGetField(oInPtr, 0, "maxIter")) != NULL){
			MATLAB_ASSERT(mxGetClassID(curField) == mxDOUBLE_CLASS, "Wrong structure type for options: expected DOUBLE for field <<maxIter>>");
			MATLAB_ASSERT(mxGetNumberOfElements(curField) == 1, "Wrong structure type for options: expected 1 number for field <<maxIter>>");
			m_iterMax = (int)(*(double*)mxGetData(curField));
			MATLAB_ASSERT(m_iterMax >= 1, "Wrong value for options.maxIter: expected value is >= 1");
		}
		if((curField = mxGetField(oInPtr, 0, "verbosity")) != NULL){
			MATLAB_ASSERT(mxGetClassID(curField) == mxDOUBLE_CLASS, "Wrong structure type for options: expected DOUBLE for field <<verbosity>>");
			MATLAB_ASSERT(mxGetNumberOfElements(curField) == 1, "Wrong structure type for options: expected 1 number for field <<verbosity>>");
			verbosityLevel = (int)(*(double*)mxGetData(curField));
			MATLAB_ASSERT(verbosityLevel == 0 || verbosityLevel == 1 || verbosityLevel == 2, "Wrong value for options.verbosity: expected value is 0, 1, or 2");
		}
		if((curField = mxGetField(oInPtr, 0, "funcEps")) != NULL){
			MATLAB_ASSERT(mxGetClassID(curField) == mxDOUBLE_CLASS, "Wrong structure type for options: expected DOUBLE for field <<funcEps>>");
			MATLAB_ASSERT(mxGetNumberOfElements(curField) == 1, "Wrong structure type for options: expected 1 number for field <<funcEps>>");
			m_eps = *(double*)mxGetData(curField);
		}
		if((curField = mxGetField(oInPtr, 0, "printMinIter")) != NULL){
			MATLAB_ASSERT(mxGetClassID(curField) == mxDOUBLE_CLASS, "Wrong structure type for options: expected DOUBLE for field <<printMinIter>>");
			MATLAB_ASSERT(mxGetNumberOfElements(curField) == 1, "Wrong structure type for options: expected 1 number for field <<printMinIter>>");
			m_printMinIter = (int)(*(double*)mxGetData(curField));
			MATLAB_ASSERT(m_printMinIter >= 0, "Wrong value for options.printMinIter: expected value is >= 0");
		}
		if((curField = mxGetField(oInPtr, 0, "printIter")) != NULL){
			MATLAB_ASSERT(mxGetClassID(curField) == mxDOUBLE_CLASS, "Wrong structure type for options: expected DOUBLE for field <<printIter>>");
			MATLAB_ASSERT(mxGetNumberOfElements(curField) == 1, "Wrong structure type for options: expected 1 number for field <<printIter>>");
			m_printIter = (int)(*(double*)mxGetData(curField));
			MATLAB_ASSERT(m_printIter >= 1, "Wrong value for options.printIter: expected value is >= 1");
		}
	}	

	// get unary potentials
	MATLAB_ASSERT(mxGetNumberOfDimensions(uInPtr) == 2, "Unary term array is not 2-dimensional");
	MATLAB_ASSERT(mxGetPi(uInPtr) == NULL, "Unary potentials should not be complex");
	
	mwSize numNodes = mxGetN(uInPtr);
	mwSize numLabels = mxGetM(uInPtr);

	MATLAB_ASSERT(numNodes >= 1, "The number of nodes is not positive");
	MATLAB_ASSERT(numLabels >= 1, "The number of labels is not positive");
	MATLAB_ASSERT(mxGetClassID(uInPtr) == mxDOUBLE_CLASS, "Expected mxDOUBLE_CLASS for input unary term argument");
	double* termW = (double*)mxGetData(uInPtr);

	//get label matrix
	double* labelMatrix = NULL;
	if(mInPtr != NULL){
		if(!mxIsEmpty(mInPtr)){
			MATLAB_ASSERT(mxGetClassID(mInPtr) == mxDOUBLE_CLASS, "Expected mxDOUBLE_CLASS for label matrix");
			MATLAB_ASSERT(mxGetNumberOfDimensions(mInPtr) == 2, "Label matrix is not 2-dimensional");
			MATLAB_ASSERT(mxGetPi(mInPtr) == NULL, "Label matrix should not be complex");
			MATLAB_ASSERT(mxGetN(mInPtr) == numLabels && mxGetM(mInPtr) == numLabels, "Label matrix should be of size NumLabels x NumLabels");
			
			labelMatrix = (double*)mxGetData(mInPtr);
		}
	}

	//get pairwise potentials
	MATLAB_ASSERT(mxIsSparse(pInPtr), "Expected sparse array for neighbours");
	MATLAB_ASSERT(mxGetN(pInPtr) == numNodes && mxGetM(pInPtr) == numNodes,
	              "Neighbours array must be NumNodes x NumNodes in size");
	MATLAB_ASSERT(mxGetClassID(pInPtr) == mxDOUBLE_CLASS, "Expected mxDOUBLE_CLASS for neighbours array");
	MATLAB_ASSERT(mxGetPi(pInPtr) == NULL, "Pairwise potentials should not be complex");

	mwIndex colNum = (mwIndex)mxGetN(pInPtr);
	const mwIndex* ir = mxGetIr(pInPtr);
	const mwIndex* jc = mxGetJc(pInPtr);
	double*        pr = mxGetPr(pInPtr);

	//create variables to store answer
	double lowerBoundMy  = 0;
	double energyMy = 0;
	vector<double> segmentMy(numNodes, 0);
	vector<double> timePlot;
	vector<double> energyPlot;
	vector<double> lbPlot;

	//create MRF object for general potentials
	if ( labelMatrix != NULL ) {

		//prepare default options
		MRFEnergy<TypeGeneral>::Options options;
		options.m_eps = m_eps; 
		options.m_iterMax = m_iterMax;
		options.m_printIter = m_printIter;     
		options.m_printMinIter = m_printMinIter;

		MRFEnergy<TypeGeneral>* mrf;
		MRFEnergy<TypeGeneral>::NodeId* nodes;
	
		TypeGeneral::REAL energy, lowerBound;
		
		TypeGeneral::REAL *D = new TypeGeneral::REAL[numLabels];
		TypeGeneral::REAL *P = new TypeGeneral::REAL[numLabels * numLabels];
		for(int i = 0; i < numLabels * numLabels; ++i)	P[i] = 0;

		mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
		nodes = new MRFEnergy<TypeGeneral>::NodeId[numNodes];
	
		// construct energy
		// add unary terms
		for(int i = 0; i < numNodes; ++i){
			nodes[i] = mrf->AddNode(TypeGeneral::LocalSize(numLabels), TypeGeneral::NodeData(termW + i * numLabels));
		}

		//add pairwise terms
		for (mwIndex c = 0; c < colNum; ++c) {
			mwIndex rowStart = jc[c]; 
			mwIndex rowEnd   = jc[c + 1]; 
			for (mwIndex ri = rowStart; ri < rowEnd; ++ri)  {
				mwIndex r = ir[ri];
				double dw = pr[ri];

				// Add a general term
				if (r < c) { // pick only upper triangle
					for(int i = 0; i < numLabels; ++i)
						for(int j = 0; j < numLabels; ++j)
							P[j + numLabels * i] = dw * labelMatrix[j + numLabels * i];
	
					mrf->AddEdge(nodes[r], nodes[c], TypeGeneral::EdgeData(TypeGeneral::GENERAL, P));
				}
			 }
		 }	

		 /////////////////////// TRW-S algorithm //////////////////////
		 if (verbosityLevel < 2)
			options.m_printMinIter = options.m_iterMax + 2;

			clock_t tStart = clock();
			
			if(method == 0) //TRW-S
			{
				// Function below is optional - it may help if, for example, nodes are added in a random order
				mrf->SetAutomaticOrdering();
	
				mrf->Minimize_TRW_S(options, lowerBound, energy);

				if(verbosityLevel >= 1)
					printf("TRW-S finished. Time: %f\n", (clock() - tStart) * 1.0 / CLOCKS_PER_SEC);
			}
			else
			{
				// Function below is optional - it may help if, for example, nodes are added in a random order
				//mrf->SetAutomaticOrdering();

				mrf->Minimize_BP(options, energy);
				lowerBound = std::numeric_limits<double>::signaling_NaN();

				if(verbosityLevel >= 1)
					printf("BP finished. Time: %f\n", (clock() - tStart) * 1.0 / CLOCKS_PER_SEC);
			}

		// save solution
		energyMy = energy;
		lowerBoundMy = lowerBound;
		for( int i = 0; i < numNodes; ++i) {
			segmentMy[i] = (double)(mrf -> GetSolution(nodes[i])) + 1;
		}
		timePlot.reserve(m_iterMax);
		energyPlot.reserve(m_iterMax);;
		lbPlot.reserve(m_iterMax);;
		for(int i = 0; i < m_iterMax; ++i) {
			double curTime = (double)(mrf -> timePlot[i]);
			if (curTime < -1e-2) break;
			
			timePlot.push_back(curTime);
			lbPlot.push_back( (double)(mrf -> lbPlot[i]) );
			energyPlot.push_back( (double)(mrf -> ePlot[i]) );
		}


		// done
		delete [] nodes;
		delete mrf;
		delete [] D;
		delete [] P;
	} else {
	// Potts MRF
		//check pairwise terms
		mwSize numEdges = 0;
		for (mwIndex c = 0; c < colNum; ++c) {
			mwIndex rowStart = jc[c]; 
			mwIndex rowEnd   = jc[c+1]; 
			for (mwIndex ri = rowStart; ri < rowEnd; ++ri)  {
				mwIndex r = ir[ri];
	
				double dw = pr[ri];
				if( r < c) numEdges++;
				if (dw < 0)
					mexErrMsgTxt("Some Potts edge have negative coefficient!");

			}
		}

		//prepare default options
		MRFEnergy<TypePotts>::Options options;
		options.m_eps = m_eps; 
		options.m_iterMax = m_iterMax;
		options.m_printIter = m_printIter;     
		options.m_printMinIter = m_printMinIter;

		MRFEnergy<TypePotts>* mrf;
		MRFEnergy<TypePotts>::NodeId* nodes;
	
		TypePotts::REAL energy, lowerBound;
		
		TypePotts::REAL *D = new TypePotts::REAL[numLabels];
		
		mrf = new MRFEnergy<TypePotts>(TypePotts::GlobalSize(numLabels));
		nodes = new MRFEnergy<TypePotts>::NodeId[numNodes];
	
		// construct energy
		// add unary terms
		for(int i = 0; i < numNodes; ++i){
			nodes[i] = mrf->AddNode(TypePotts::LocalSize(), TypePotts::NodeData(termW + i * numLabels));
		}

		//add pairwise terms
		for (mwIndex c = 0; c < colNum; ++c) {
			mwIndex rowStart = jc[c]; 
			mwIndex rowEnd   = jc[c + 1]; 
			for (mwIndex ri = rowStart; ri < rowEnd; ++ri)  {
				mwIndex r = ir[ri];
				double dw = pr[ri];

				if (r < c) {
					if (dw >= 0) {
						mrf->AddEdge(nodes[r], nodes[c], TypePotts::EdgeData(dw));
					}
				}
			 }
		 }	

		 /////////////////////// TRW-S algorithm //////////////////////
		 if (verbosityLevel < 2)
			options.m_printMinIter = options.m_iterMax + 2;

			clock_t tStart = clock();
			
			if(method == 0) //TRW-S
			{
				// Function below is optional - it may help if, for example, nodes are added in a random order
				mrf->SetAutomaticOrdering();
	
				mrf->Minimize_TRW_S(options, lowerBound, energy);

				if(verbosityLevel >= 1)
					printf("TRW-S finished. Time: %f\n", (clock() - tStart) * 1.0 / CLOCKS_PER_SEC);
			}
			else
			{
				// Function below is optional - it may help if, for example, nodes are added in a random order
				//mrf->SetAutomaticOrdering();

				mrf->Minimize_BP(options, energy);
				lowerBound = std::numeric_limits<double>::signaling_NaN();

				if(verbosityLevel >= 1)
					printf("BP finished. Time: %f\n", (clock() - tStart) * 1.0 / CLOCKS_PER_SEC);
			}

		// save solution
		energyMy = energy;
		lowerBoundMy = lowerBound;
		for( int i = 0; i < numNodes; ++i) {
			segmentMy[i] = (double)(mrf -> GetSolution(nodes[i])) + 1;
		}
		timePlot.reserve(m_iterMax);
		energyPlot.reserve(m_iterMax);;
		lbPlot.reserve(m_iterMax);;
		for(int i = 0; i < m_iterMax; ++i) {
			double curTime = (double)(mrf -> timePlot[i]);
			if (curTime < -1e-2) break;
			
			timePlot.push_back(curTime);
			lbPlot.push_back( (double)(mrf -> lbPlot[i]) );
			energyPlot.push_back( (double)(mrf -> ePlot[i]) );
		}


		// done
		delete [] nodes;
		delete mrf;
		delete [] D;
	}

	//output the best energy value
	if(eOutPtr != NULL)	{
		*eOutPtr = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
		*(double*)mxGetData(*eOutPtr) = (double)energyMy;
	}

	//output the best solution
	if(sOutPtr != NULL)	{
		*sOutPtr = mxCreateNumericMatrix(numNodes, 1, mxDOUBLE_CLASS, mxREAL);
		double* segment = (double*)mxGetData(*sOutPtr);
		for(int i = 0; i < numNodes; ++i)
			segment[i] = segmentMy[i];
	}

	//output the best lower bound
	if(lbOutPtr != NULL)	{
		*lbOutPtr = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
		*(double*)mxGetData(*lbOutPtr) = (double)lowerBoundMy;
	}

	//output time plot
	if(timePlotOutPtr != NULL)	{
		*timePlotOutPtr = mxCreateNumericMatrix(timePlot.size(), 1, mxDOUBLE_CLASS, mxREAL);
		double* time = (double*)mxGetData(*timePlotOutPtr);
		for(int i = 0; i < timePlot.size(); ++i)
			time[i] = timePlot[i];
	}

	//output lower bound plot
	if(lbPlotOutPtr != NULL)	{
		*lbPlotOutPtr = mxCreateNumericMatrix(lbPlot.size(), 1, mxDOUBLE_CLASS, mxREAL);
		double* lb = (double*)mxGetData(*lbPlotOutPtr);
		for(int i = 0; i < lbPlot.size(); ++i)
			lb[i] = lbPlot[i];
	}

	//output energy plot
	if(energyPlotOutPtr != NULL)	{
		*energyPlotOutPtr = mxCreateNumericMatrix(energyPlot.size(), 1, mxDOUBLE_CLASS, mxREAL);
		double* curE = (double*)mxGetData(*energyPlotOutPtr);
		for(int i = 0; i < energyPlot.size(); ++i)
			curE[i] = energyPlot[i];
	}


}

