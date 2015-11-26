%cropRectanglesMex crops multiple bounding boxes from the initial image and resizes them to the standard output size.
% The operation is performed on a GPU using NVIDIA Performance Primitives (NPP) library
% cropRectanglesMex was created to prepare batches for training CNNs using MatConvNet (http://www.vlfeat.org/matconvnet/).
% 
% Usage:
% crops = cropRectanglesMex( im, boundingBoxes, outputSize);
% 	
% Inputs:
% im  - the image to crop from, should be a 3 channel image (dimension order: height, width, channels) of type single. 
%        Normalization (e.g. [0,1] or [0, 255]) is not important. The image should be stored in RAM (not GPU).
% boundingBoxes - bounding boxes to crop, double[ numBoundingBoxes x 4 ], each line corresponds to one bounding box. 
%       The bounding box format is y1, x1, y2, x2, where the origin is in the top-left corner. 
%       Pixels are indexed starting from 1 (e.g. [1 1 2 2] corresponds to the box containing the 4 top-left pixels of the image).
%       Bounding boxes can be partially outside of the image. The default value for filling such areas is 0 in all the channels.
% outputSize - the target size of the resized crops, double[2 x 1]. outputSize(1) - the height, outputSize(2) - the width.
% 
% Outputs:
% crops - the cropped and resized patches, gpuArray, single[ outputSize(1), outputSize(2), numChannels = 3, numBoundingBoxes ]
%
% The function can be compiled using build_cropResizeMex.m. 
% example_cropRectanglesMex.m provides the example of usage

% Anton Osokin, firstname.lastname@gmail.com, March 2015

