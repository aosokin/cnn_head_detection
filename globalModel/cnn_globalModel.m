function cnn_globalModel(varargin )
%cnn_globalModel runs the fulls instance of the CNN training procedure for the global model

if ~exist('varargin', 'var')
    varargin = {};
end

%% parse parameters
opts = struct;
opts.networkInitialization = []; % handle to the function initializing the network
opts.expDir = '';
opts.randSeed = 1;
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
opts.train.batchSize = 32; % number of images to form a batch
opts.train.numEpochs = 3;
opts.train.learningRate = [0.0001 0.00001 0.000001];
opts.train.weightDecay = 0.0005;
opts.train.backPropDepth = +inf;
opts.train.expDir = '';
opts.train.numValidationPerEpoch = 2;
opts.train.restartEpoch = nan;
opts.train.conserveMemory = true;
opts.train.continue = false;

% training batch generation
%opts.getBatch.randSeed = 1;
opts.getBatch.jitterStd = 0;
opts.getBatch.grid_size = [1 2 4 8];

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
    fprintf('Reading imdb file %s\n', opts.imdbPath);
    imdb = load(opts.imdbPath);
    if isfield(imdb, 'opts') && isequal( imdb.opts, opts.dataset )
        generateDataFlag = false;
    else
        generateDataFlag = true;
        warning('imdb file is not consistent with the parameters, need to generate it again');
    end
end
if generateDataFlag
    fprintf('Generating imdb file %s\n', opts.imdbPath);
    imdb = cnn_prepareData_localModel( opts.dataset, 'dataPath', opts.dataPath );
    imdb.opts = opts.dataset;
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb', '-v7.3') ;
end

%% random seed
% fix random seed here as well to exclude the effect of changing the seed while generating data
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
    error('cnn_globalModel:noInitNetwork', 'Initialization not provided');
end
    
opts.getBatch.meanImage = net.normalization.averageImage;
%opts.getBatch.randStream = RandStream('mt19937ar','Seed',opts.getBatch.randSeed);
opts.getBatch.dataPath = opts.dataPath;
    
%% batch generator
batchWrapper = @(imdb, batch) cnn_getBatch_globalModel(imdb, batch, opts.getBatch) ;
    
%% training
cnn_train_globalModel(net, imdb, batchWrapper, ...
   opts.train, ...
   'train', find(imdb.images.set == 1), ...
   'val', find(imdb.images.set == 2) );
end
