function imdb = cnn_prepareData_localModel( varargin )
%cnn_prepareData_localModel prepares the dataset for training the local model

if ~exist('varargin', 'var')
    varargin = {};
end

%% parse parameters
opts = struct;
opts.dataPath = '';
opts.trainingSetFile = '';
opts.validationSetFile = '';
opts.testSetFile = '';
opts.groundTruthLocalPrefix = '';
opts.imageLocalPrefix = '';
opts.candidateLocalPrefix = '';

opts.numThreads = 4;

% parse input
opts = vl_argparse(opts, varargin);

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

fprintf('Reading annotation for %d images\n', numImages);
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

end


