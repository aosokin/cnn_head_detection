
#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <npp.h>
#include <cuda_runtime.h>
#include <Exceptions.h>
#include <helper_cuda.h>

#include <math.h> 

#define MATLAB_ASSERT(expr,msg) if (!(expr)) { mexErrMsgTxt(msg);}

#if !defined(MX_API_VER) || MX_API_VER < 0x07030000
typedef size_t mwSize;
typedef size_t mwIndex;
#endif

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	MATLAB_ASSERT( nrhs == 3, "cropRectanglesMex: Wrong number of input parameters: expected 3");
    MATLAB_ASSERT( nlhs == 1, "cropRectanglesMex: Wrong number of output arguments: expected 1");
	
	// Fix input parameter order:
	const mxArray *imInPtr = (nrhs >= 0) ? prhs[0] : NULL; // image
	const mxArray *bbInPtr = (nrhs >= 1) ? prhs[1] : NULL; // bounding boxes
	const mxArray *szInPtr = (nrhs >= 2) ? prhs[2] : NULL; // output image size
	
	// Fix output parameter order:
	mxArray **cropsOutPtr = (nlhs >= 1) ? &plhs[0] : NULL; // croped and resized patches
	
	// Get the image
	MATLAB_ASSERT(mxGetNumberOfDimensions(imInPtr) == 3, "cropRectanglesMex: the image is not 3-dimensional");
	MATLAB_ASSERT(mxGetClassID(imInPtr) == mxSINGLE_CLASS, "cropRectanglesMex: the image should be of type SINGLE");
	MATLAB_ASSERT(mxGetPi(imInPtr) == NULL, "cropRectanglesMex: image should not be complex");

    const mwSize* dimensions = mxGetDimensions(imInPtr);
	mwSize imageHeight = dimensions[0];
	mwSize imageWidth = dimensions[1];
	mwSize numChannels = dimensions[2];
	MATLAB_ASSERT(numChannels == 3, "cropRectanglesMex: image should contain 3 channels");

	float* imageData = (float*) mxGetData(imInPtr);

	// get bounding boxes
	MATLAB_ASSERT(mxGetNumberOfDimensions(bbInPtr) == 2, "cropRectanglesMex: <boundingBoxes> input is not 2-dimensional");
	MATLAB_ASSERT(mxGetClassID(bbInPtr) == mxDOUBLE_CLASS, "cropRectanglesMex: <boundingBoxes> input is not of type double");
	MATLAB_ASSERT(mxGetPi(bbInPtr) == NULL, "cropRectanglesMex: <boundingBoxes> input should not be complex");
	MATLAB_ASSERT(mxGetN(bbInPtr) == 4, "cropRectanglesMex: <boundingBoxes> input should be of size #boundingBoxes x 4");
	
	mwSize numBb = mxGetM(bbInPtr);
	double* bbData = (double*) mxGetData(bbInPtr); // y1, x1, y2, x2

	// get output size
	MATLAB_ASSERT(mxGetNumberOfElements(szInPtr) == 2, "cropRectanglesMex: <outputSize> input should contain 2 numbers");
	MATLAB_ASSERT(mxGetClassID(szInPtr) == mxDOUBLE_CLASS, "cropRectanglesMex: <outputSize> input is not of type double");
	MATLAB_ASSERT(mxGetPi(szInPtr) == NULL, "cropRectanglesMex: <outputSize> input should not be complex");
	
	double* outputSizeData = (double*) mxGetData(szInPtr);
	int targetHeight = (int) (outputSizeData[0] + 0.5);
	int targetWidth = (int) (outputSizeData[1] + 0.5);

	// initialize GPU
	mxInitGPU();

	// copy image to the GPU
	mxGPUArray const *inputImage;
    float const *d_inputImage;
	inputImage = mxGPUCreateFromMxArray(imInPtr);
	d_inputImage = (float const *)(mxGPUGetDataReadOnly(inputImage));

	// allocate memory for the output
    mxGPUArray *outputData;
    float *d_outputData;
	const mwSize outputDimensions[4] = { targetHeight, targetWidth, numChannels, numBb };
	outputData = mxGPUCreateGPUArray(4, outputDimensions, mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES) ; //MX_GPU_DO_NOT_INITIALIZE);
	d_outputData = (float *)(mxGPUGetData(outputData));

	// initialize some cropping arguments
	NppiSize nppiImageSize = {};
	nppiImageSize.width = imageHeight; // CAUTION: NPPI thinks that the image is transposed 
	nppiImageSize.height = imageWidth;

	int channelValueSize = sizeof(float);
	int imageStep = imageHeight * channelValueSize;
	int targetStep = targetHeight * channelValueSize;

    NppiRect targetRect = {};
    targetRect.x = 0;
    targetRect.y = 0;
    targetRect.width = targetHeight;
    targetRect.height = targetWidth;
	
	// the main loop over bounding boxes
	for(int iBb = 0; iBb < numBb; ++iBb) {

		double y1 = bbData[ iBb ] - 1;
		double x1 = bbData[ iBb + numBb ] - 1;
		double y2 = bbData[ iBb + numBb * 2 ] - 1;
		double x2 = bbData[ iBb + numBb * 3 ] - 1;

	    double nXFactor = double( targetHeight ) / ( y2 - y1 + 1 );
		double nYFactor = double( targetWidth ) / ( x2 - x1 + 1 );
		double nXShift = -nXFactor * (double(y1) + 0.5) + 0.5;
		double nYShift = -nYFactor * (double(x1) + 0.5) + 0.5;

		NppiRect sourceRect = {};
    	sourceRect.x = (int) floor(y1);
    	sourceRect.y = (int) floor(x1);
    	sourceRect.width =  (int) ceil(y2 - y1 + 1);
    	sourceRect.height = (int) ceil(x2 - x1 + 1);
    	if (sourceRect.width <= 1) {
    		sourceRect.width = 2;
    	}
    	if (sourceRect.height <= 1) {
    		sourceRect.height = 2;
    	}

    	// adjust bounding box bounds if it is outside of the image
    	if (sourceRect.x < 0) {
    		sourceRect.width = sourceRect.width + sourceRect.x;
    		sourceRect.x = 0.0;
    	}
    	if (sourceRect.y < 0) {
    		sourceRect.height = sourceRect.height + sourceRect.y;
    		sourceRect.y = 0.0;
    	}
    	if (sourceRect.width > imageHeight - sourceRect.x + 1) {
    		sourceRect.width = imageHeight - sourceRect.x + 1;
    	}
    	if (sourceRect.height > imageWidth - sourceRect.y + 1) {
    		sourceRect.height = imageWidth - sourceRect.y + 1;
    	}

    	float *curOutput = d_outputData + numChannels * targetHeight * targetWidth * iBb;
		const float *pSrc[3] = { d_inputImage, d_inputImage + imageHeight * imageWidth, d_inputImage + 2 * imageHeight * imageWidth};
		float *pDst[3] = { curOutput, curOutput + targetHeight * targetWidth, curOutput + 2 * targetHeight * targetWidth};

		NPP_CHECK_NPP( nppiResizeSqrPixel_32f_P3R (
			pSrc,  // const Npp32f *pSrc, 
			nppiImageSize, // nppiSize oSrcSize, 
			imageStep, // int nSrcStep, 
			sourceRect, // NppiRect oSrcROI, 
			pDst, // Npp8u *pDst, 
			targetStep, // int nDstStep, 
			targetRect, // NppiRect oDstROI, 
			nXFactor, nYFactor, nXShift, nYShift, 
			NPPI_INTER_CUBIC //int eInterpolation
			) );
}
	
	*cropsOutPtr = mxGPUCreateMxArrayOnGPU(outputData);

	// do not forget to free GPU memory
	mxGPUDestroyGPUArray(outputData);
	mxGPUDestroyGPUArray(inputImage);
}
