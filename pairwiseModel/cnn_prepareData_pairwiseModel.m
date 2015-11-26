function imdb = cnn_prepareData_pairwiseModel( varargin )
%cnn_prepareData_pairwiseModel prepares the dataset for training the pairwise model

if ~exist('varargin', 'var')
    varargin = {};
end

%% parse parameters
opts = struct;
% same as in the local model:
opts.dataPath = '';
opts.trainingSetFile = '';
opts.validationSetFile = '';
opts.testSetFile = '';
opts.groundTruthLocalPrefix = '';
opts.imageLocalPrefix = '';
opts.candidateLocalPrefix = '';

% specific to the pairwise model:
opts.scoreFile = [];
opts.maxNumPatchesPerImage = 32;
opts.nmsIntersectionOverAreaThreshold = 0.3;
opts.numPairwiseClusters = 20;
opts.clusterInfo = struct('type', [], 'mean', [], 'std', [], 'numClusters', [], 'clusterCenters', [] );
opts.clusterFunction = [];

opts.randomSeed = 1;
opts.numThreads = 4;

% parse input
opts = vl_argparse(opts, varargin);

if isempty(opts.scoreFile)
    error( 'cnn_prepareData_pairwiseModel:noScoreFile', 'No score file specified' );
end
if ~exist(opts.scoreFile, 'file')
    error( 'cnn_prepareData_pairwiseModel:scoreFileNotFound', ['Score file not found: ', opts.scoreFile] );
end

%% get images and candidates
fileLists = { opts.trainingSetFile; opts.validationSetFile; opts.testSetFile };
fileNames = cell( numel( fileLists ), 1 );
numImages = 0;
for iList = 1 : numel( fileLists )
    if ~isempty( fileLists{iList} )
        fileNames{iList} = readLines( fullfile( opts.dataPath, fileLists{iList} ) );
        numImages = numImages + numel( fileNames{iList} );
    end
end
imageFiles = cell( numImages, 1 );
candidateFiles = cell( numImages, 1 );
groundTruthFiles = cell( numImages, 1 );
globalImageCount = 0;
for iList = 1 : numel( fileLists )
    if ~isempty( fileLists{iList} )
        for iImage = 1 : numel( fileNames{iList} )
            globalImageCount = globalImageCount + 1;
            imageFiles{globalImageCount} = fullfile( opts.imageLocalPrefix, [fileNames{iList}{iImage}, '.jpeg'] );
            candidateFiles{globalImageCount} = fullfile( opts.candidateLocalPrefix, [fileNames{iList}{iImage}, '.mat'] );
            groundTruthFiles{globalImageCount} = fullfile( opts.groundTruthLocalPrefix, [fileNames{iList}{iImage}, '.xml'] );
        end
    end
end

%% setup the imdb structure
imdb = struct;
imdb.imageFiles = imageFiles;
imdb.candidateFiles = candidateFiles;
imdb.groundTruthFiles = groundTruthFiles;
imdb.images.set = zeros(numImages, 1);
imdb.images.set( 1 : numel(fileNames{1}) ) = 1; % training set
imdb.images.set( numel(fileNames{1}) + 1 : numel(fileNames{1}) + numel(fileNames{2}) ) = 2; % validation set
imdb.images.set( numel(fileNames{1}) + numel(fileNames{2}) + 1 : numel(fileNames{1}) + numel(fileNames{2}) + numel(fileNames{3}) ) = 3; % test set

%% get GT
groundTruth = cell(numImages, 1);
isDifficultGroundTruth = cell(numImages, 1);
groundTruthFiles = imdb.groundTruthFiles;

tStart = tic;
fprintf('Reading annotation for %d images...\n', numImages);
parfor (iImage = 1 : numImages, opts.numThreads)
    if mod( iImage, 1000 ) == 0
        fprintf( 'Image %d\n', iImage )
    end
    
    annotation = VOCreadrecxml( fullfile( opts.dataPath, groundTruthFiles{iImage} ) );
    if isfield(annotation, 'objects')
        groundTruth{iImage} = nan( numel( annotation.objects ), 4 );
        isDifficultGroundTruth{iImage} = false( numel( annotation.objects ), 1 );
        
        for iGt = 1 : numel( annotation.objects )
            groundTruth{iImage}(iGt, :) = annotation.objects(iGt).bbox; % bbox is in X1 Y1 X2 Y2 format
            isDifficultGroundTruth{iImage}(iGt) = annotation.objects(iGt).difficult;
        end
    else
        % there is no annotation for this image
        groundTruth{iImage} = nan( 0, 4 );
        isDifficultGroundTruth{iImage} = false( 0, 1 );
    end
end
imdb.groundTruth = groundTruth;
imdb.isDifficultGroundTruth = isDifficultGroundTruth;
fprintf('Reading time: %fs\n', toc(tStart) );

%% get all image sizes
fprintf('Getting image sizes ... ');
tStart = tic;
imageFiles = imdb.imageFiles;
imageSizes = cell( numImages, 1 );
curPrefix = opts.dataPath;
parfor (iImage = 1 : numImages, opts.numThreads)
    curInfo = imfinfo( fullfile( curPrefix, imageFiles{iImage} ) );
    imageSizes{iImage} = [ curInfo.Height, curInfo.Width ];
end
imdb.images.size = imageSizes;
fprintf('%fs\n', toc(tStart) );

%% get initial scores for candidate preselection
[ imdb.scores, imdb.candidateIds ] = pruneCandidatesNms( imdb, opts.scoreFile, ...
    'maxNumPatchesPerImage',  opts.maxNumPatchesPerImage, ...
    'nmsIntersectionOverAreaThreshold',  opts.nmsIntersectionOverAreaThreshold, ...
    'dataPath', opts.dataPath, ...
    'numThreads', opts.numThreads );

%% get clusterings
if isempty( opts.clusterInfo ) || isempty( opts.clusterFunction )
    tStart = tic;
    fprintf('Clustering pairs of the training set ... \n');
    [~, clusterInfo] =  clusterCandidatePairs_3logFeatures_normalized_kmeans(imdb, find(imdb.images.set == 1), ...
        'numClusters', opts.numPairwiseClusters, ...
        'dataPath', opts.dataPath, ...
        'randomSeed', opts.randomSeed );
    imdb.clusterInfo = clusterInfo;
    imdb.clusterFunction = @clusterCandidatePairs_3logFeatures_normalized_kmeans;
    
    fprintf( 'Clustering time: %fs\n', toc(tStart) );
else
    fprintf('Using the user provided clustering\n');
    imdb.clusterInfo = opts.clusterInfo;
    imdb.clusterFunction = opts.clusterFunction;
end


end


