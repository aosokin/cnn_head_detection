%run_training_localModel is the launching script for the training of the local model

% SETUP THESE PATHS TO RUN THE CODE
dataPath = 'data/HollywoodHeads';
resultPath = 'results/HollywoodHeads';
pretrainedNetworkPath = 'models';

%% add all the required paths
filePath = fileparts( mfilename('fullpath') );
run( fullfile( fileparts( filePath ), 'setup.m' ) );

%% set data files
% network initialization
pretrainedNetwork = fullfile( pretrainedNetworkPath, 'imagenet-torch-oquab.mat'); networkInputSize = [224, 224]; initNetworkName = 'oquabTorch'; addDropout = false;
%CAUTION: the MatConvNet pretrained networks do not have dropout layers! be careful when turning this option on!
%pretrainedNetwork = fullfile( pretrainedNetworkPath, 'imagenet-caffe-alex.mat'); networkInputSize = [227, 227]; initNetworkName = 'alexCaffe'; addDropout = true;
%pretrainedNetwork = fullfile( pretrainedNetworkPath, 'imagenet-vgg-s.mat'); networkInputSize = [227, 227]; initNetworkName = 'vggS'; addDropout = true;
%pretrainedNetwork = fullfile( pretrainedNetworkPath, 'imagenet-vgg-verydeep-16.mat'); networkInputSize = [227, 227]; initNetworkName = 'vggVeryDeep16'; addDropout = true;

% get the mean vector on the training set
meanVector = [57, 52, 47];

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

opts_cnn.expDir =  resultPath;
opts_cnn.maxGpuImagesEvaluation = 3000;
opts_cnn.scoreMode = 'afterSoftMax'; % 'beforeSoftMax' or 'afterSoftMax' or 'scoreDifference';

opts_cnn.train.numValidationPerEpoch = 8;
opts_cnn.train.numEpochs = 3;
opts_cnn.train.learningRate =  [ 0.001 0.0001 0.00001 ];
opts_cnn.train.batchSize = 1; 
opts_cnn.train.backPropDepth = +inf;
opts_cnn.train.weightDecay = 0.0005;

opts_cnn.getBatch.numPatchesPerImage = 64;
opts_cnn.getBatch.maxPositives = 32;
opts_cnn.getBatch.iouPositiveThreshold = 0.6;
opts_cnn.getBatch.iouNegativeThreshold = 0.5;
opts_cnn.getBatch.jitterStd = 1;
opts_cnn.getBatch.cropPad = [18, 18, 18, 18];

opts_cnn.evaluation.iouThreshold = 0.5;
opts_cnn.evaluation.useDifficultImages = false;

opts_cnn.train.expDir = fullfile( opts_cnn.expDir, 'local', 'models');

% get the mean image for normalization
meanImage = single( repmat( reshape( meanVector, [1 1 3] ), networkInputSize ) );

% network initialization
extraLayers = [];
opts_cnn.networkInitialization = @() cnn_initNet_localModel( pretrainedNetwork, 2, meanImage, extraLayers, addDropout );

%% run training
cnn_localModel( opts_cnn );




