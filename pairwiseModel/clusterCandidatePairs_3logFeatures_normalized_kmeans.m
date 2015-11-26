function [clusteredEdges, clusterInfo] = clusterCandidatePairs_3logFeatures_normalized_kmeans(imdb, imageSet, varargin)
%clusterCandidatePairs_3logFeatures_normalized_kmeans assigns all pairs of candidates to clusters, k-means clustering is performed if necessary

if ~exist('varargin', 'var')
    varargin = {};
end
%% parameters
opts = struct;
opts.numClusters = 10;
opts.clusterInfo = struct('type', [], 'mean', [], 'std', [], 'numClusters', [], 'clusterCenters', [] );
opts.maxPointsToCluster = 10^6;
opts.labelIouThreshold = 0.4;
opts.scoreThreshold = 0;
opts.dataPath = '';
opts.candidateIds = [];
opts.randomSeed = 1;
opts = vl_argparse(opts, varargin);

if ~exist('imageSet', 'var') || isempty( imageSet )
    error( 'clusterCandidatePairs_3logFeatures_normalized_kmeans:emptyImageSet', 'There are not images to do clustering' );
end

%% get the candidate data
dataset = struct;
dataset.boundingBoxes = cell(length(imageSet), 1);
dataset.numCandidates = zeros(length(imageSet), 1);
dataset.numCandidatesInitial = zeros(length(imageSet), 1);
dataset.groundTruth = cell(length(imageSet), 1);
dataset.scores = cell(length(imageSet), 1);
dataset.imageSize = cell(length(imageSet), 1);

numBoxPairs = 0;

selectedCandidatesIds = cell(length(imageSet), 1);
for iImageId = 1 : length(imageSet)
    if mod( iImageId, 1000 ) == 0
        fprintf('Reading data for image %d of %d\n', iImageId, length(imageSet));
    end
    
    iImage = imageSet( iImageId );
    
    curCandidates = load( fullfile( opts.dataPath, imdb.candidateFiles{ iImage } ) );
    
    dataset.numCandidatesInitial(iImageId) = size( curCandidates.boxes, 1);
    
    selectedCandidatesIds{iImageId} = [];
    if isempty( opts.candidateIds )
        selectedCandidatesIds{iImageId} = imdb.candidateIds{ iImage };
    else
        selectedCandidatesIds{iImageId} = opts.candidateIds{iImageId};
    end
    curCandidates = curCandidates.boxes( selectedCandidatesIds{iImageId}, :);
    
    % fix the Bb format: SelectiveSearch format [y1 x1 y2 x2] to format [x y w h]
    curBb = convertBb_Y1X1Y2X2_to_X1Y1WH( curCandidates(:, 1 : 4) );
    
    % fix the ground-truth format
    curGt = convertBb_X1Y1X2Y2_to_X1Y1WH( imdb.groundTruth{iImage} );
   
    dataset.imageSize{iImageId} = imdb.images.size{iImage};
    dataset.numCandidates(iImageId) = size(curBb, 1);
    dataset.boundingBoxes{iImageId} = curBb;
    dataset.groundTruth{iImageId} = curGt;
    dataset.scores{iImageId} = imdb.scores{iImage};
    
    numBoxPairs = numBoxPairs + size(curBb, 1) * (size(curBb, 1) - 1) / 2;
end


%% get info
numObjects = length(dataset.numCandidates);
numBoxFeatures = 8;
bBoxData = nan(numBoxPairs, numBoxFeatures);
bBoxPairLabels = nan(numBoxPairs, 2);
bBoxPairScores = nan(numBoxPairs, 2);
image_w = nan(numBoxPairs, 1);
image_h = nan(numBoxPairs, 1);
imageId = nan(numBoxPairs, 1);

clusteredEdges = cell(numObjects, 1);
startObjectId = zeros(numObjects + 1, 1);

%% collect all pairs
iPair = 0;
for iObject = 1 : numObjects
    startObjectId(iObject) = iPair + 1;
    curBbNum = dataset.numCandidates(iObject);
    curPairNum = curBbNum * (curBbNum - 1) / 2;
    curIds = iPair + 1 : iPair + curPairNum;
    iPair = iPair + curPairNum;
    
    [bb1, bb2] = meshgrid(1 : curBbNum, 1 : curBbNum);
    
    curMask = bb1 < bb2;
    bb1 = bb1(curMask);
    bb1 = bb1(:);
    bb2 = bb2(curMask);
    bb2= bb2(:);
    
    curBBoxes = dataset.boundingBoxes{ iObject };
    
    curGtIoU = zeros( size(curBBoxes, 1), 1 );
    for iGt = 1 : size( dataset.groundTruth{iObject}, 1 )
        IoU = bbIntersectionOverUnion( curBBoxes, dataset.groundTruth{iObject}(iGt, :) );
        curGtIoU = max( curGtIoU, IoU );
    end
    curLabels = curGtIoU > opts.labelIouThreshold;
    
    bBoxData(curIds, :) = [ curBBoxes( bb1, : ), curBBoxes( bb2, : ) ];
    bBoxPairLabels(curIds, :) = [ curLabels(bb1), curLabels(bb2) ];
    bBoxPairScores(curIds, :) = [ dataset.scores{iObject}(bb1), dataset.scores{iObject}(bb2) ];
    
    image_w( curIds ) = dataset.imageSize{iObject}( 2 );
    image_h( curIds ) = dataset.imageSize{iObject}( 1 );
    imageId( curIds ) = iObject;
    
    clusteredEdges{iObject} = struct;
    clusteredEdges{iObject}.bbIds = [bb1, bb2];
    clusteredEdges{iObject}.clusterId = nan(size(bb1, 1), 1);
end
startObjectId( numObjects + 1 ) = iPair + 1;
bBoxData(iPair + 1 : end, :) = [];
image_w(iPair + 1 : end, :) = [];
image_h(iPair + 1 : end, :) = [];
bBoxPairLabels(iPair + 1 : end, :) = [];
bBoxPairScores(iPair + 1 : end, :) = [];
imageId(iPair + 1 : end, :) = [];

%% sort bouding boxes (left to right)
changeOrder = ( bBoxData(:, 1) > bBoxData(:, 5) ) | ( (bBoxData(:, 1) == bBoxData(:, 5)) & (bBoxData(:, 2) > bBoxData(:, 6)) );
tmp = bBoxData(changeOrder, 1 : 4);
bBoxData(changeOrder, 1 : 4) = bBoxData(changeOrder, 5 : 8);
bBoxData(changeOrder, 5 : 8) = tmp;

tmp = bBoxPairLabels(changeOrder, 1);
bBoxPairLabels(changeOrder, 1) = bBoxPairLabels(changeOrder, 2);
bBoxPairLabels(changeOrder, 2) = tmp;

tmp = bBoxPairScores(changeOrder, 1);
bBoxPairScores(changeOrder, 1) = bBoxPairScores(changeOrder, 2);
bBoxPairScores(changeOrder, 2) = tmp;


%% get BB features
cx1 = bBoxData(:, 1) + bBoxData(:, 3) / 2;
cx2 = bBoxData(:, 5) + bBoxData(:, 7) / 2;
cy1 = bBoxData(:, 2) + bBoxData(:, 4) / 2;
cy2 = bBoxData(:, 6) + bBoxData(:, 8) / 2;
w1 = bBoxData(:, 3);
w2 = bBoxData(:, 7);
h1 = bBoxData(:, 4);
h2 = bBoxData(:, 8);
s1 = (w1 + h1) / 2;
s2 = (w2 + h2) / 2;

f1 = s1 ./ s2;
f2 = (cx2 - cx1) ./ s1;
f3 = (cy2 - cy1) ./ s1;
f1 = log(f1);
f2 = log( abs(f2) + 1 ) .* sign(f2);
f3 = log( abs(f3) + 1 ) .* sign(f3);
features = [f1, f2, f3];

%% clustering
if isempty( opts.clusterInfo.type )         
    %% Run kmeans
    rng(opts.randomSeed , 'twister');
    
    % normalize the features
    meanFeatures = mean( features, 1 );
    stdFeatures = std( features, 0, 1 );
    
    featuresNormalized = bsxfun( @minus, features, meanFeatures );
    featuresNormalized = bsxfun( @rdivide, featuresNormalized, stdFeatures );
    
    clusterInfo = struct;
    clusterInfo.type = '3logFeatures_normalized';
    clusterInfo.mean = meanFeatures;
    clusterInfo.std = stdFeatures;
    clusterInfo.numClusters = opts.numClusters;
    clusterInfo.clusterCenters = [];
    
    
    curMask = ( ( bBoxPairScores(:, 1) > opts.scoreThreshold ) | ( bBoxPairLabels(:, 1) == 1) ) ...
        & ( ( bBoxPairScores(:, 2) > opts.scoreThreshold ) | ( bBoxPairLabels(:, 2) == 1) );
    
    curIds = find( curMask );
    if numel( curIds ) > opts.maxPointsToCluster
        curIds = curIds( randperm( numel( curIds ), opts.maxPointsToCluster ) );
    end
    
    [~,  clusterInfo.clusterCenters] = kmeans( featuresNormalized(curIds, :), opts.numClusters );
   
else
    clusterInfo = opts.clusterInfo;
    featuresNormalized = bsxfun( @minus, features, clusterInfo.mean );
    featuresNormalized = bsxfun( @rdivide, featuresNormalized, clusterInfo.std );
end
clusterIds = assignPointsToClusters( clusterInfo.clusterCenters, featuresNormalized );

%% change order within some pairs bounding boxes, revert to original indices
for iObject = 1 : numObjects
    curIds = startObjectId(iObject) : 1 : startObjectId(iObject + 1) - 1;
    changeFlags = changeOrder( curIds );
    curClusterIds = clusterIds( curIds );

    tmp = clusteredEdges{iObject}.bbIds(changeFlags, 1); 
    clusteredEdges{iObject}.bbIds(changeFlags, 1) = clusteredEdges{iObject}.bbIds(changeFlags, 2);
    clusteredEdges{iObject}.bbIds(changeFlags, 2) = tmp;
    clusteredEdges{iObject}.clusterId = curClusterIds;
    
    clusteredEdges{iObject}.initialBbIds = selectedCandidatesIds{ iObject }( clusteredEdges{iObject}.bbIds );
end


end

























