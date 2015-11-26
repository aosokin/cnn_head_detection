function [cropsResized, preparationTime, resizeTime] = cropImagePatches(curImage, boundingBoxes, cropPad, outputSize, maxGpuImages, cropMode, jitterStd)
%cropImagePatches crops patches from an image and resizes them to the standard size
%
% cropsResized = cropImagePatches(curImage, boundingBoxes, cropPad, outputSize);
% [cropsResized, cropTime, resizeTime] = cropImagePatches(curImage, boundingBoxes, cropPad, outputSize, maxGpuImages, cropMode, jitterStd);
%
% Input:
%   cropImage - image to crop from (has to be single)
%   boundingBoxes - bounding boxes in format [y1 x1 y2 x2]. x is col, y is row 
%   cropPad - padding of the bounding boxes (padding is measure after crop and resize), format: left, top, right, bottom
%   outputSize - target size of the crops
%   maxGpuImages - maximal number of patches to crop on a GPU at the same time (default : 256)
%   cropMode - 'square' or 'warp' (mimicing the R-CNN cropping code) (default: 'warp')
%   jitterStd - ammount of jittering to do: std of a gaussian distribution of a patch-border shift (default: 0)
%
% Output:
%   cropsResized - the cropped patches
%   preparationTime - time on the computations of the crop parameters
%   resizeTime - time spent on cropping and resizing
%
% This function depends on cropRectanglesMex

tStart = tic;

if ~exist( 'maxGpuImages', 'var') || isempty(maxGpuImages)
    maxGpuImages = 256;
end
if ~exist( 'cropMode', 'var') || isempty(cropMode)
    cropMode = 'warp';
end
if ~exist('jitterStd', 'var') || isempty(jitterStd)
    jitterStd = 0;
end
useSquare = false;
if isequal( cropMode, 'square')
    useSquare = true;
end

numChannels = size(curImage, 3);
numBoxes = size( boundingBoxes, 1 );
if numBoxes <= maxGpuImages
    resultOnGpuGlag = true;
else
    resultOnGpuGlag = false;
end

cropBoxes = nan(numBoxes, 4);
for iBox = 1 : numBoxes
    %% get crop position: take padding into account
    leftBorder = boundingBoxes(iBox, 2);
    rightBorder = boundingBoxes(iBox, 4);
    
    topBorder = boundingBoxes(iBox, 1);
    bottomBorder = boundingBoxes(iBox, 3);
    
    % add the square mode
    if useSquare
        halfWidth = (rightBorder - leftBorder + 1 ) / 2;
        halfHeight = (bottomBorder - topBorder + 1 ) / 2;
        
        centerX = leftBorder + halfWidth;
        centerY = topBorder + halfHeight;
        
        if halfHeight > halfWidth
            halfWidth = halfHeight;
        else
            halfHeight = halfWidth;
        end
        
        topBorder = centerY - halfHeight;
        bottomBorder = centerY + halfHeight;
        leftBorder = centerX - halfWidth;
        rightBorder = centerX + halfWidth;
    end
    
    % compute the transformation
    posOld = [  topBorder, leftBorder, 1; ... % top left corner
        topBorder, rightBorder, 1; ... % top right corner
        bottomBorder, leftBorder, 1; ... % bottom left corner
        bottomBorder, rightBorder, 1; ... % bottom right corner
        ]';
    
    % crop corners after padding
    posNew = [  1 + cropPad(2), 1 + cropPad(1), 1; ... % top left corner
        1 + cropPad(2), outputSize(2) - cropPad(3), 1; ... % top right corner
        outputSize(1) - cropPad(4), 1 + cropPad(1), 1; ... % bottom left corner
        outputSize(1) - cropPad(4), outputSize(2)- cropPad(3), 1; ... % bottom right corner
        ]';
    
    % solve linear system
    M = posOld / posNew;

    % actual crop corners
    posNew = [  1, 1, 1; ... % top left corner
        1, outputSize(2), 1; ... % top right corner
        outputSize(1), 1, 1; ... % bottom left corner
        outputSize(1), outputSize(2), 1; ... % bottom right corner
        ]';
    
    cropPos = (M * posNew)';
    
    leftBorder = (cropPos(1, 2) + cropPos(3, 2)) / 2;
    rightBorder = (cropPos(2, 2) + cropPos(4, 2)) / 2;
    
    topBorder = (cropPos(1, 1) + cropPos(2, 1)) / 2;
    bottomBorder = (cropPos(3, 1) + cropPos(4, 1)) / 2;
    
    cropBoxes(iBox, 1) = topBorder;
    cropBoxes(iBox, 2) = leftBorder;
    cropBoxes(iBox, 3) = bottomBorder;
    cropBoxes(iBox, 4) = rightBorder;
    
end

%% apply jittering
borderNoise = randn( size( cropBoxes) ) * jitterStd;
cropBoxes = cropBoxes + borderNoise;
% check if the bouding boxes are still valid. If not (might happen if the size of the box is too small) than remove jittering
badMask = cropBoxes(:, 1) > cropBoxes(iBox, 3) | cropBoxes(:, 2) > cropBoxes(iBox, 4);
cropBoxes( badMask, : ) = cropBoxes( badMask, : ) - borderNoise( badMask, : );

preparationTime = toc(tStart);

%% crop the image
if ~isequal( class( curImage ), 'single')
    curImage = single(curImage);
end
if resultOnGpuGlag
    cropsResized = cropRectanglesMex( curImage, cropBoxes, outputSize );
    cropsResized = single(cropsResized);
else
    cropsResized = zeros(outputSize(1), outputSize(2), numChannels, numBoxes, 'single');
    for iBatchStart = 1 : maxGpuImages : numBoxes
        curIds = iBatchStart : 1 : min( numBoxes, iBatchStart + maxGpuImages - 1);
        curBoxes = cropBoxes( curIds, : );
        curCrops = cropRectanglesMex( curImage, curBoxes, outputSize );
        cropsResized(:,:,:,curIds) = gather(curCrops);
    end
    cropsResized = single(cropsResized);
end

resizeTime = toc(tStart) - preparationTime;

end
