%run_training_pairwiseModel is the launching script for the training of the pairwise model

% SETUP THESE PATHS TO RUN THE CODE
pretrainedNetworkPath = 'data/matconvnet';
dataPath = 'data/HollywoodHeads';
resultPath = 'results/HollywoodHeads';

%% add all the required paths
filePath = fileparts( mfilename('fullpath') );
run( fullfile( fileparts( filePath ), 'setup.m' ) );

% network initialization
initNetworkFile = fullfile( 'models', 'local.mat' ); 
initNetworkName = 'oquab';

% scores to preselect candidates
candidateSelectionScoresFile = fullfile( resultPath, 'local', 'localModel-scores-trainValTest.mat' );

% pairwise clusters
if exist( fullfile(resultPath, 'pairwise', 'imdb_pairwise_precomputedClusters.mat'), 'file' )
    % use the precomputed clusters
    clusters = load( fullfile(resultPath, 'pairwise', 'imdb_pairwise_precomputedClusters.mat'), 'clusterInfo', 'clusterFunction' );
else
    % compute the clustering on the fly
    clusters = struct;
    clusters.clusterInfo = struct('type', [], 'mean', [], 'std', [], 'numClusters', [], 'clusterCenters', [] );
    clusters.clusterFunction = [];
end

%% preparation
% parameters
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

opts_cnn.loss = 'logisticScoresCompact';
opts_cnn.expDir =  resultPath;
opts_cnn.scoreMode = 'maxMarginals';

opts_cnn.train.batchSize = 4;
opts_cnn.train.numEpochs = 8;
opts_cnn.train.learningRate = [ 0.00001*ones(1, 4), 0.000001*ones(1, 4) ];
opts_cnn.train.weightDecay = 0.0005 / 100;
opts_cnn.train.numValidationPerEpoch = 1;
opts_cnn.train.backPropagateType = 'unaryAndPairwise'; % 'all', 'unaryAndPairwise', 'onlyUnary', 'onlyPairwise'
opts_cnn.train.disableDropoutFeatureExtractor = false;

opts_cnn.getBatch.cropMode = 'warp';
opts_cnn.getBatch.jitterStd = 1;
opts_cnn.getBatch.iouPositiveNegativeThreshold = 0.5;
opts_cnn.getBatch.randomizeCandidates = false;
opts_cnn.getBatch.nmsIntersectionOverAreaThreshold = 0.3;
opts_cnn.getBatch.cropPad = [18, 18, 18, 18];

opts_cnn.evaluation.iouThreshold = 0.5;
opts_cnn.evaluation.useDifficultImages = false;

opts_cnn.networkInitialization = @() cnn_initNet_pairwiseModel( initNetworkFile, opts_cnn.loss, opts_cnn.dataset.numPairwiseClusters );

opts_cnn.train.expDir = fullfile( opts_cnn.expDir, 'pairwise', 'models');

%% run training
cnn_pairwiseModel( opts_cnn );
