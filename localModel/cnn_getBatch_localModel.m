function [im, labels, patchInfo] = cnn_getBatch_localModel(imdb, batch, varargin)
%cnn_getBatch_localModel constructs the batch to train the local model

tVeryStart = tic;

%% parse parameters
opts = struct;
opts.numPatchesPerImage = 64;
opts.maxPositives = 32;
opts.cropPad = [18 18 18 18];
opts.meanImage = nan;
opts.iouPositiveThreshold = 0.4;
opts.iouNegativeThreshold = 0.4;
opts.randStream = RandStream('mt19937ar','Seed',1);
opts.numThreads = 4;
opts.maxGpuImages = 256; % maximal number of patches to crop on a GPU at the same time
opts.aspectRatioThreshold = 1.5;
opts.cropMode = 'warp'; % 'square' or 'warp'
opts.randSeed = nan;
opts.jitterStd = 0;
opts.dataPath = '';
opts = vl_argparse(opts, varargin);

if isnan(opts.meanImage)
    error('cnn_getBatch_localModel:noMeanImage', 'mean image parameter is not specified');
end
opts.afterCropSize = [ size(opts.meanImage, 1), size(opts.meanImage, 2) ];

if isinf( opts.numPatchesPerImage ) && length(batch) > 1
    error('cnn_getBatch_localModel:wrongMode', 'extracting unlimited number of patches is possible for one image only');
end

cropTime = 0;
resizeTime = 0;

tStart = tic;

%% extract data
numImages = length(batch);
% batchImages = cell(numImages, 1);
batchCandidates = cell(numImages, 1);
batchGt = cell(numImages, 1);
batchImages = cell(numImages, 1);

% get full paths for the batch images
jpegImagesFlag = true;
curImageNames = cell( numel(batch), 1 );
for iImage = 1 : numel(batch)
    curImageNames{iImage} = fullfile( opts.dataPath, imdb.imageFiles{ batch(iImage) } );
    
    % check if all the images are in JPEG format
    [~,~,curExt] = fileparts( curImageNames{iImage} );
    if ~isequal( curExt, '.jpeg') && ~isequal( curExt, '.jpg')
        jpegImagesFlag = false;
    end
end

% read the images
if jpegImagesFlag
    batchImages = vl_imreadjpeg( curImageNames, 'NumThreads', opts.numThreads);
else
    for iImage = 1 : numImages
        batchImages{iImage} = single( imread( curImageNames{iImage} ) );
    end
end
% read other data
for iImage = 1 : numImages
    batchCandidates{iImage} = load( fullfile( opts.dataPath, imdb.candidateFiles{ batch(iImage) } ) );
    batchGt{iImage} = imdb.groundTruth{ batch(iImage) };
end
readTime = toc(tStart);

%% create batch
if ~isinf( opts.numPatchesPerImage )
    im = zeros( opts.afterCropSize(1), opts.afterCropSize(2), 3, opts.numPatchesPerImage * numImages, 'single', 'gpuArray' );
    labels = zeros( opts.numPatchesPerImage * numImages, 1 );
else
    im = zeros( opts.afterCropSize(1), opts.afterCropSize(2), 3, 0, 'single', 'gpuArray' );
    labels = [];
end

candidateIds = cell(numImages, 1);
patchOffset = 0;
for iImage = 1 : numImages
    curImage = single( batchImages{iImage} );
    %    curGroundTruth = batchGt{iImage};
    
    randOrder = randperm( opts.randStream, size(batchCandidates{iImage}.boxes, 1) );
    curCandidates = batchCandidates{iImage}.boxes(randOrder, :);
    
    % filter out candidates with too small or too large aspect ratio
    aspectRatio = (curCandidates(:, 3) - curCandidates(:, 1) + 1) ./ (curCandidates(:, 4) - curCandidates(:, 2) + 1);
    minAspectRatio = min( opts.aspectRatioThreshold, 1 / opts.aspectRatioThreshold );
    maxAspectRatio = max( opts.aspectRatioThreshold, 1 / opts.aspectRatioThreshold );
    badCandidatesMask = aspectRatio >= maxAspectRatio | aspectRatio <= minAspectRatio;
    curCandidates( badCandidatesMask, :) = [];
    candidateIds{iImage} = randOrder;
    candidateIds{iImage}( badCandidatesMask ) = [];
    
    % compute the Intersection-over-Union scores
    bestIou = zeros( numel(candidateIds{iImage}), 1 );
    curCandidates_X1Y1WH = convertBb_Y1X1Y2X2_to_X1Y1WH( curCandidates(:, 1:4) );
    curGt_X1Y1WH = convertBb_X1Y1X2Y2_to_X1Y1WH( batchGt{iImage} );
    for iGt = 1 : size( batchGt{iImage}, 1)
        curIou = bbIntersectionOverUnion( curCandidates_X1Y1WH, curGt_X1Y1WH(iGt, :) );
        bestIou = max( bestIou, curIou );
    end
    positiveCandidates = find( bestIou >= opts.iouPositiveThreshold );
    negativeCandidates = find( bestIou < opts.iouNegativeThreshold );
    
    if length( positiveCandidates ) > opts.maxPositives
        positiveCandidates = positiveCandidates(1 : opts.maxPositives);
    end
    negativeCandidates = negativeCandidates(1 : min( opts.numPatchesPerImage - length( positiveCandidates ), length(negativeCandidates) ) );
    
    toCropCandidates = [positiveCandidates(:); negativeCandidates(:)];
    curLabels = [ones(numel(positiveCandidates),1); 2 * ones(numel(negativeCandidates),1)];
    
    curPatchIds = patchOffset + ( 1 : length(toCropCandidates) );
    patchOffset = patchOffset + length(toCropCandidates);
    
    candidateIds{iImage} = candidateIds{iImage}( toCropCandidates );
    toCropPatches = curCandidates( toCropCandidates, 1 : 4 );
    
    % % visualize bounding boxes
    % boxImage = showBoundingBoxes(batchImages{iImage},  [convertBb_Y1X1Y2X2_to_X1Y1WH( toCropPatches ); curGt_X1Y1WH], ...
    %     [ repmat({'r'}, [size(toCropPatches,1), 1]); repmat({'y'}, [size(curGt_X1Y1WH,1), 1])] );
    % imshow( boxImage )
    
    % crop patches
    if ~isempty(im)
        [im(:, :, :, curPatchIds), curCropTime, curResizeTime] = cropImagePatches( curImage, toCropPatches, opts.cropPad, opts.afterCropSize, opts.maxGpuImages, opts.cropMode, opts.jitterStd );
        labels( curPatchIds ) = curLabels;
    else
        [im, curCropTime, curResizeTime] = cropImagePatches( curImage, toCropPatches, opts.cropPad, opts.afterCropSize, opts.maxGpuImages, opts.cropMode, opts.jitterStd );
        labels = curLabels;
    end
    cropTime = cropTime + curCropTime;
    resizeTime = resizeTime + curResizeTime;
end
im( :, :, :, patchOffset + 1 : end ) = [];
im = bsxfun(@minus, im, opts.meanImage);
labels = labels(:);
labels( patchOffset + 1 : end ) = [];

patchInfo = struct;
patchInfo.readTime = readTime;
patchInfo.cropTime = cropTime;
patchInfo.resizeTime = resizeTime;
patchInfo.batchTime = toc(tVeryStart);
patchInfo.wasteTime = patchInfo.batchTime - readTime - cropTime - resizeTime;
patchInfo.candidateIds = candidateIds;

end

