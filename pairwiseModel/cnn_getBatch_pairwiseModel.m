function [im, labels, patchInfo] = cnn_getBatch_pairwiseModel(imdb, batch, varargin)
%cnn_getBatch_pairwiseModel constructs the batch to train the pairwise model

tVeryStart = tic;

%% parse parameters
opts = struct;
opts.cropPad = [18 18 18 18];
opts.meanImage = nan;
opts.iouPositiveNegativeThreshold = 0.5;
opts.randStream = RandStream('mt19937ar','Seed',1);
opts.numThreads = 4;
opts.maxGpuImages = inf;
opts.cropMode = 'warp'; % 'warp' or 'square'
opts.jitterStd = 0;
opts.dataPath = '';
opts.randomizeCandidates = false; % select the candidates not based on the precomputed scores but randomly
opts.nmsIntersectionOverAreaThreshold = 0.3; % only active if opts.randomizeCandidates == true
opts.randSeed = nan;
opts = vl_argparse(opts, varargin);

if isnan(opts.meanImage)
    error('cnn_getBatch_pairwiseModel:noMeanImage', 'the mean image parameter is not specified');
end
opts.afterCropSize = [ size(opts.meanImage, 1), size(opts.meanImage, 2) ];

readTime = 0;
cropTime = 0;
resizeTime = 0;

tStart = tic;

%% extract data
numImages = length(batch);
% batchImages = cell(numImages, 1);
batchCandidates = cell(numImages, 1);
batchGt = cell(numImages, 1);
batchScores = cell(numImages, 1);
batchCandidateIds = cell(numImages, 1);
batchImages = cell(numImages, 1);

curImageNames = imdb.imageFiles( batch );

% check if all the images are in JPEG format
jpegImagesFlag = true;
for iImage = 1 : numImages
    [~,~,curExt] = fileparts( curImageNames{iImage} );
    if ~isequal( curExt, '.jpeg') && ~isequal( curExt, '.jpg')
        jpegImagesFlag = false;
    end
    curImageNames{iImage} = fullfile( opts.dataPath, curImageNames{iImage} );
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
numPatchesJoint = 0;
for iImage = 1 : numImages
    % get bounding boxes, be careful with the formats!
    readCandidates = load( fullfile( opts.dataPath, imdb.candidateFiles{ batch(iImage) } ) );
    batchCandidates{iImage} = convertBb_Y1X1Y2X2_to_X1Y1WH( readCandidates.boxes );
    batchGt{iImage} = convertBb_X1Y1X2Y2_to_X1Y1WH( imdb.groundTruth{ batch(iImage) } );
    
    if opts.randomizeCandidates
        % generate random scores to do NMS
        % it is probably not important how to sample: The scores have to be in [0,1] bacause later 1 is added to the positive candidates
        randScores = rand(opts.randStream, [size( batchCandidates{iImage}, 1 ), 1] ); 
    
        % make scores for positives bigger than scores for negatives
        if ~isempty(batchGt{iImage})
            gtOverlap = false( size( batchCandidates{iImage}, 1 ), 1);
            for iGt  = 1 : size( batchGt{iImage}, 1 )
                curIou = bbIntersectionOverUnion( batchCandidates{iImage}, batchGt{iImage}(iGt, :) );
                gtOverlap = gtOverlap | curIou > opts.iouPositiveNegativeThreshold;
            end
            randScores( gtOverlap ) = randScores( gtOverlap ) + 1;
        end
    
        % do NMS 
        idsNms = selectBoundingBoxesNonMaxSup( batchCandidates{iImage}, randScores, ...
            'numBoundingBoxMax', numel( imdb.candidateIds{ batch(iImage) } ), ...
            'nmsIntersectionOverAreaThreshold', opts.nmsIntersectionOverAreaThreshold );

        batchScores{iImage} = randScores(idsNms, :);    
        batchCandidateIds{iImage} = idsNms(:);
    else
        batchScores{iImage} = imdb.scores{ batch(iImage) }(:);
        batchCandidateIds{iImage} = imdb.candidateIds{ batch(iImage) }(:);
    end
    
    numPatchesJoint = numPatchesJoint + numel( batchCandidateIds{iImage} );
end
readTime = toc(tStart);

% %% visualize bounding boxes
% iImage = 1;
% boxImage = showBoundingBoxes(batchImages{iImage}, batchCandidates{iImage}(batchCandidateIds{iImage}, :), 'r');
% imshow( boxImage )

%% create batch
im = zeros( opts.afterCropSize(1), opts.afterCropSize(2), 3, numPatchesJoint, 'single', 'gpuArray' );
labels = cell(numImages, 1);

numPairwiseClusters = imdb.clusterInfo.numClusters;
clusteredEdges = imdb.clusterFunction( imdb, batch, ...
    'clusterInfo', imdb.clusterInfo, ...
    'numClusters', numPairwiseClusters, ...
    'dataPath', opts.dataPath, ...
    'candidateIds', batchCandidateIds );

cropedPatchesNumber = 0;
for iImage = 1 : numImages
    curImage = single( batchImages{iImage} );
    curCandidates = batchCandidates{iImage}(batchCandidateIds{iImage}, :);
    curGroundTruth = batchGt{iImage};
    
    % crop patches
    [curIm, curCropTime, curResizeTime] = cropImagePatches( curImage, convertBb_X1Y1WH_to_Y1X1Y2X2(curCandidates), opts.cropPad, opts.afterCropSize, opts.maxGpuImages, opts.cropMode, opts.jitterStd );
    cropTime = cropTime + curCropTime;
    resizeTime = resizeTime + curResizeTime;
    numCropped = size(curIm, 4);
    curCropIds = cropedPatchesNumber + (1 : numCropped);
    im( :,:,:,curCropIds ) = curIm;
    cropedPatchesNumber = cropedPatchesNumber + numCropped;
    
    % prepare labels
    labels{iImage}.candidateBatchIds = curCropIds;
    %labels{iImage}.candidateScores = curScores;
    labels{iImage}.classGroundTruth = zeros(numCropped, 1);
    labels{iImage}.instanceGroundTruth = zeros(numCropped, 1);
    
    % assign candidates to the best matching ground-truth objects
    if ~isempty(curGroundTruth)
        for iCrop  = 1 : numCropped
            curIou = bbIntersectionOverUnion( curGroundTruth, curCandidates(iCrop, :) );
            [maxIou, bestMatch] = max(curIou);
            if maxIou > opts.iouPositiveNegativeThreshold
                labels{iImage}.instanceGroundTruth(iCrop) = bestMatch;
            end
        end
    else
        labels{iImage}.instanceGroundTruth = zeros(numCropped, 1);
    end
      
    % compare the matches with the labels
    for iGt = 1 : size(curGroundTruth, 1)
        goodCandidates = find(  labels{iImage}.instanceGroundTruth == iGt );
        if numel(goodCandidates) > 0
            bestMatch = goodCandidates( randi(opts.randStream, [1, numel(goodCandidates)] ) );
            labels{iImage}.classGroundTruth(bestMatch) = 1;
        end
    end
    
    % save edges
    labels{iImage}.clusteredEdges = clusteredEdges{iImage};
end
im = im(:,:,:,1 : cropedPatchesNumber);
im = bsxfun(@minus, im, opts.meanImage);

patchInfo = struct;
patchInfo.readTime = readTime;
patchInfo.cropTime = cropTime;
patchInfo.resizeTime = resizeTime;
patchInfo.batchTime = toc(tVeryStart);
patchInfo.wasteTime = patchInfo.batchTime - readTime - cropTime - resizeTime;
patchInfo.candidateIds = batchCandidateIds;

end

