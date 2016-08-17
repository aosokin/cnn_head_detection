%run_computeScores_pairwiseModel applies the pairwise model to the bounding-box proposals to compute their scores

% SETUP THESE PATHS TO RUN THE CODE
dataPath = 'data/HollywoodHeads';
resultPath = 'results/HollywoodHeads';

% network to evaluate
netFile = fullfile( 'models', 'pairwise.mat' );

% file to store the scores
resultFile = fullfile( resultPath, 'pairwise', 'pairwiseModel-scores-test.mat' );

% Choose subset of data to compute the scores.
%   To run the evaluation of the pairwise model you need the test test (3).
scoreSubset = 3; % 1 - train subset, 2 - validation, 3 - test; can do [1,2,3] to compute scores on all the subsets

%% setup paths
filePath = fileparts( mfilename('fullpath') );
run( fullfile( fileparts( filePath ), 'setup.m' ) );

% scores to preselect candidates
candidateSelectionScoresFile = fullfile( resultPath, 'local', 'localModel-scores-trainValTest.mat' );
if ~exist( candidateSelectionScoresFile, 'file' ) && isequal( scoreSubset, 3 )
    candidateSelectionScoresFile = fullfile( resultPath, 'local', 'localModel-scores-test.mat' );
end

% pairwise clusters
if exist( fullfile(resultPath, 'pairwise', 'imdb_pairwise_precomputedClusters.mat'), 'file' )
    % use the precomputed clusters
    clusters = load( fullfile(resultPath, 'pairwise', 'imdb_pairwise_precomputedClusters.mat'), 'clusterInfo', 'clusterFunction' );
else
    % compute the clustering on the fly
    clusters = struct;
    clusters.clusterInfo = struct('type', [], 'mean', [], 'std', [], 'numClusters', [], 'clusterCenters', [] );
    clusters.clusterFunction = [];
    % CAUTION: you will need score of the local model on the training set to do this operation
    candidateSelectionScoresFile = fullfile( resultPath, 'local', 'localModel-scores-trainValTest.mat' );
end

%% parameters
opts_cnn = struct;
opts_cnn.dataPath = dataPath;
opts_cnn.dataset.trainingSetFile = fullfile('Splits', 'train.txt');
opts_cnn.dataset.validationSetFile = fullfile('Splits', 'val.txt');
opts_cnn.dataset.testSetFile = fullfile('Splits', 'test.txt');
opts_cnn.dataset.groundTruthLocalPrefix = 'Annotations';
opts_cnn.dataset.imageLocalPrefix = 'JPEGImages';
opts_cnn.dataset.candidateLocalPrefix = 'Candidates';
opts_cnn.dataset.scoreFile = candidateSelectionScoresFile;
opts_cnn.dataset.maxNumPatchesPerImage = 16;
opts_cnn.dataset.nmsIntersectionOverAreaThreshold = 0.3;
opts_cnn.dataset.numPairwiseClusters = 20;
opts_cnn.dataset.clusterInfo = clusters.clusterInfo;
opts_cnn.dataset.clusterFunction = clusters.clusterFunction;

opts_cnn.expDir =  resultPath;
opts_cnn.scoreMode = 'maxMarginals';

opts_cnn.evaluation.iouThreshold = 0.5;
opts_cnn.evaluation.useDifficultImages = false;

%% load dataset
opts_cnn.imdbPath = fullfile( opts_cnn.expDir, 'imdb_pairwise.mat' );
if exist(opts_cnn.imdbPath, 'file')
    fprintf('Reading imdb file %s\n', opts_cnn.imdbPath);
    imdb = load(opts_cnn.imdbPath);
    if isfield(imdb, 'opts') && isequal( imdb.opts, opts_cnn.dataset )
        fprintf('imdb.opts is compatible with the opt_cnn.dataset\n')
    else
        warning('opts_cnn.dataset parameters are not compatible with the provided imdb file. Be careful!');
    end
else
    warning('imdb file is not found. Making it will take some time.');
    fprintf('Generating imdb file %s\n', opts_cnn.imdbPath);
    imdb = cnn_prepareData_pairwiseModel( opts_cnn.dataset, 'dataPath', opts_cnn.dataPath );
    imdb.opts = opts_cnn.dataset;
    mkdir(opts_cnn.expDir);
    save(opts_cnn.imdbPath, '-struct', 'imdb', '-v7.3') ;
end

%% prepare the network
net = load( netFile, '-mat');
if ~isfield(net, 'layers') && isfield(net, 'net')
    net = net.net;
end
if ~isfield(net, 'layers')
    error('Could not load the network!');
end
net = vl_simplenn_move(net, 'gpu');

%% select the set of images to run evaluation
imageSetToDoPr = false( numel( imdb.images.set ), 1 );
for iSubset = 1 : numel( scoreSubset ) 
    imageSetToDoPr = imageSetToDoPr | ( imdb.images.set(:) == scoreSubset( iSubset ) );
end
imageSetToDoPr = find( imageSetToDoPr );

%% start the evaluation
opts_cnn.getBatchEvaluation = struct;
opts_cnn.getBatchEvaluation.meanImage = net.normalization.averageImage;
opts_cnn.getBatchEvaluation.jitterStd = 0;
opts_cnn.getBatchEvaluation.iouPositiveNegativeThreshold = opts_cnn.evaluation.iouThreshold;
opts_cnn.getBatchEvaluation.dataPath = opts_cnn.dataPath;
opts_cnn.getBatchEvaluation.randomizeCandidates = false;
opts_cnn.getBatch.cropMode = 'warp';
opts_cnn.getBatch.cropPad = [18, 18, 18, 18];

batchWrapperEvaluation = @(imdb, batch) cnn_getBatch_pairwiseModel(imdb, batch, opts_cnn.getBatchEvaluation);

[scores, candidateIds] = cnn_computeScores_pairwiseModel( net, imdb, batchWrapperEvaluation, ...
    'imageSet', imageSetToDoPr, ...
    'scoreMode', opts_cnn.scoreMode );

if exist(resultFile, 'file')
    warning('The results file already exists. Overwriting!');
end
if ~exist(fileparts(resultFile), 'dir')
    mkdir(fileparts(resultFile));
end
save( resultFile, 'scores', 'candidateIds', '-v7.3' );

%% save detections to files
detSavePath = fullfile( resultPath, 'pairwise', 'dets');
if ~exist(detSavePath, 'dir')
    mkdir(detSavePath)
end
detSaveFormat = fullfile(detSavePath, '%s.mat');
disp('Saving detection files');
for i=1:length(imageSetToDoPr)
    imgIdx = imageSetToDoPr(i);
    cand_path = fullfile(dataPath, imdb.candidateFiles{imgIdx});
    load(cand_path);
    %load candidate
    BB = [convertBb_X1Y1X2Y2_to_X1Y1WH(boxes(candidateIds{imgIdx},[2 1 4 3])) scores{imgIdx}];
    [~, im_name, ~] = fileparts(imdb.imageFiles{imgIdx});
    savePath = sprintf(detSaveFormat, im_name);
    save(savePath, 'BB');
end