function cnn_localModel( varargin )
%cnn_localModel runs the fulls instance of the CNN training procedure for the local model

if ~exist('varargin', 'var')
    varargin = {};
end

%% parse parameters
opts = struct;
opts.networkInitialization = []; % handle to the function initializing the network
opts.scoreMode = 'beforeSoftMax'; % 'beforeSoftMax' or 'afterSoftMax' or 'scoreDifference';
opts.expDir = ''; % folder to store the results of network training
opts.randSeed = 1;
opts.maxGpuImagesEvaluation = 256; % maximal number of patches to crop on a GPU at the same time
opts.dataPath = '';
opts.imdbPath = ''; % path to the file with the dataset information created by cnn_prepareData_localModel.m; default: fullfile(opts.expDir, 'imdb.mat')

% dataset info
opts.dataset = struct;
opts.dataset.trainingSetFile = '';
opts.dataset.validationSetFile = '';
opts.dataset.testSetFile = '';
opts.dataset.groundTruthLocalPrefix = '';
opts.dataset.imageLocalPrefix = '';
opts.dataset.candidateLocalPrefix = '';

% CNN training
opts.train.batchSize = 1; % number of images to form a batch
opts.train.numEpochs = 3;
opts.train.learningRate = [ 0.001 0.0001 0.00001 ];
opts.train.weightDecay = 0.0005;
opts.train.backPropDepth = +inf;
opts.train.expDir = '';
opts.train.numValidationPerEpoch = 8;
opts.train.restartEpoch = nan;
opts.train.conserveMemory = true;

% training batch generation
opts.getBatch.numPatchesPerImage = 64;
opts.getBatch.maxPositives = 32;
opts.getBatch.randSeed = 1;
opts.getBatch.iouPositiveThreshold = 0.5;
opts.getBatch.iouNegativeThreshold = 0.6;
opts.getBatch.cropMode = 'warp';
opts.getBatch.jitterStd = 0;
opts.getBatch.cropPad = [18 18 18 18];

% evaluation parameters
opts.evaluation.iouThreshold = 0.5;
opts.evaluation.useDifficultImages = false;

% parse input
opts = vl_argparse(opts, varargin);

if isempty( opts.imdbPath )
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
end

%% random seed
% the CPU random seed
cpu_rs = RandStream('mt19937ar','Seed',opts.randSeed);
RandStream.setGlobalStream(cpu_rs);
% the GPU random seed
gpu_rs = parallel.gpu.RandStream('CombRecursive','Seed',opts.randSeed);
parallel.gpu.RandStream.setGlobalStream(gpu_rs);

%% Prepare data
generateDataFlag = true;
if exist(opts.imdbPath, 'file')
    fprintf('Reading prepared dataset file\n');
    imdb = load(opts.imdbPath);
    if isfield(imdb, 'opts') && isequal( imdb.opts, opts.dataset )
        generateDataFlag = false;
    else
        generateDataFlag = true;
        warning('Dataset parameters are not consistent with the file, need to regenerate');
    end
end
if generateDataFlag
    fprintf('Preparing the dataset\n');
    imdb = cnn_prepareData_localModel( opts.dataset, 'dataPath', opts.dataPath );
    imdb.opts = opts.dataset;
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb', '-v7.3') ;
end

%% random seed
% fix the random seeds here as well to exclude the effect of changing the seed while generating data
% the CPU random seed
cpu_rs = RandStream('mt19937ar','Seed',opts.randSeed);
RandStream.setGlobalStream(cpu_rs);
% the GPU random seed
gpu_rs = parallel.gpu.RandStream('CombRecursive','Seed',opts.randSeed);
parallel.gpu.RandStream.setGlobalStream(gpu_rs);


%% Initialize network
if ~isempty(opts.networkInitialization)
    net = opts.networkInitialization();
else
    error('cnn_localModel:noInitNetwork', 'Initialization not provided');
end

opts.getBatch.meanImage = net.normalization.averageImage;
opts.getBatch.randStream = RandStream('mt19937ar','Seed',opts.getBatch.randSeed);
opts.getBatch.dataPath = opts.dataPath;


%% batch generator
batchWrapper = @(imdb, batch) cnn_getBatch_localModel(imdb, batch, opts.getBatch ) ;

%% training
cnn_train_localModel(net, imdb, batchWrapper, ...
    opts.train, ...
    'train', find(imdb.images.set == 1), ...
    'val', find(imdb.images.set == 2) );

end
