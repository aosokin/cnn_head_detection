%run_training_globalModel is the launching script fot the experiments with the global model

% SETUP THESE PATHS TO RUN THE CODE
pretrainedNetworkPath = 'models';
dataPath = 'data/HollywoodHeads';
resultPath = 'results/HollywoodHeads';

if ~exist(resultPath, 'dir')
    mkdir(resultPath);
end

%% add all the required paths
filePath = fileparts( mfilename('fullpath') );
run( fullfile( fileparts( filePath ), 'setup.m' ) );

%% set data files
% network initialization
pretrainedNetwork = fullfile( pretrainedNetworkPath, 'imagenet-torch-oquab.mat'); networkInputSize = [224, 224]; initNetworkName = 'torch-oquab'; addDropout = false;

% get the mean image of the training set
meanVector = [57, 52, 47];

% parameters
opts_cnn = struct;
opts_cnn.dataPath = dataPath;
opts_cnn.dataset.trainingSetFile = fullfile('Splits', 'train.txt');
opts_cnn.dataset.validationSetFile = fullfile('Splits', 'val.txt');
opts_cnn.dataset.testSetFile = fullfile('Splits', 'test.txt');
opts_cnn.dataset.groundTruthLocalPrefix = 'Annotations';
opts_cnn.dataset.imageLocalPrefix = 'JPEGImages';
opts_cnn.dataset.candidateLocalPrefix = 'Candidates';

opts_cnn.expDir =  resultPath;

opts_cnn.train.numValidationPerEpoch = 2;
opts_cnn.train.numEpochs = 6;
opts_cnn.train.learningRate = [0.0001 0.0001 0.00001 0.00001 0.000001 0.000001];
opts_cnn.train.continue = true;
opts_cnn.train.batchSize = 32; 
opts_cnn.train.backPropDepth = +inf;
opts_cnn.train.weightDecay = 0.0005;

opts_cnn.train.expDir = fullfile( opts_cnn.expDir, 'global', 'models');

% get the mean image for normalization
meanImage = single( repmat( reshape( meanVector, [1 1 3] ), networkInputSize ) );

% network initialization
extraLayers = [];
opts_cnn.networkInitialization = @() cnn_initNet_globalModel( pretrainedNetwork, 284, 2, meanImage, extraLayers, addDropout );

%% run training
cnn_globalModel(opts_cnn );
