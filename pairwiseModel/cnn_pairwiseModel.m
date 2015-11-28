function cnn_pairwiseModel( varargin)
%cnn_pairwiseModel runs the fulls instance of the CNN training procedure for the pairwise model

if ~exist('varargin', 'var')
    varargin = {};
end

%% parse parameters
opts = struct;
opts.networkInitialization = [];
opts.scoreMode = 'maxMarginals'; % 'maxMarginals' - the only implemented option
opts.expDir = ''; % folder to store the results of network training
opts.randomSeed = 1;
opts.loss = 'logisticScoresCompact'; % 'logisticScoresCompact' or 'svmStructCompact'
opts.dataPath = '';
opts.imdbPath = ''; % path to the file with the dataset information created by cnn_prepareData_pairwiseModel; default: fullfile(opts.expDir, 'imdb_pairwise.mat')

% dataset info
opts.dataset = struct;
% same as in the local model:
opts.dataset.trainingSetFile = '';
opts.dataset.validationSetFile = '';
opts.dataset.testSetFile = '';
opts.dataset.groundTruthLocalPrefix = '';
opts.dataset.imageLocalPrefix = '';
opts.dataset.candidateLocalPrefix = '';
% specific to the pairwise model:
opts.dataset.scoreFile = [];
opts.dataset.maxNumPatchesPerImage = 16;
opts.dataset.nmsIntersectionOverAreaThreshold = 0.3;
opts.dataset.numPairwiseClusters = 20;
opts.dataset.clusterInfo = struct('type', [], 'mean', [], 'std', [], 'numClusters', [], 'clusterCenters', [] );
opts.dataset.clusterFunction = [];

% CNN training
opts.train.batchSize = 4; % number of images to form a batch
opts.train.numEpochs = 8;
opts.train.learningRate = [ 0.00001*ones(1, 4), 0.000001*ones(1, 4) ];
opts.train.weightDecay =  0.0005 / 100;
opts.train.backPropagateType = 'all'; % 'all', 'unaryAndPairwise', 'onlyUnary', 'onlyPairwise'
opts.train.expDir = fullfile( opts.expDir, 'pairwiseModel' );
opts.train.numValidationPerEpoch = 2;
opts.train.conserveMemory = true;
opts.train.disableDropoutFeatureExtractor = false;

% training batch generation
opts.getBatch.cropMode = 'warp';
opts.getBatch.jitterStd = 1;
opts.getBatch.iouPositiveNegativeThreshold = 0.5;
opts.getBatch.randomizeCandidates = false;  % select the candidates not based on the precomputed scores but randomly
opts.getBatch.nmsIntersectionOverAreaThreshold = 0.3; % only active if opts.randomizeCandidates == true
opts.getBatch.cropPad = [18, 18, 18, 18];
opts.getBatch.randSeed = 1;

% evaluation parameters
opts.evaluation.iouThreshold = 0.5;
opts.evaluation.useDifficultImages = false;

% parse input
opts = vl_argparse(opts, varargin);

if isempty( opts.imdbPath )
    opts.imdbPath = fullfile(opts.expDir, 'imdb_pairwise.mat');
end



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
    imdb = cnn_prepareData_pairwiseModel( opts.dataset, 'dataPath', opts.dataPath );
    imdb.opts = opts.dataset;
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb', '-v7.3') ;
end

%% random seed for the learning process
% the CPU random seed
cpu_rs = RandStream('mt19937ar','Seed',opts.randomSeed);
RandStream.setGlobalStream(cpu_rs);
% the GPU random seed
gpu_rs = parallel.gpu.RandStream('CombRecursive','Seed',opts.randomSeed);
parallel.gpu.RandStream.setGlobalStream(gpu_rs);

%% Initialize network
if ~isempty( opts.networkInitialization )
    net = opts.networkInitialization();
else
    error('cnn_pairwiseModel:noInitNetwork', 'Initialization not provided');
end

opts.getBatch.meanImage = net.normalization.averageImage;
opts.getBatch.randStream = RandStream('mt19937ar','Seed', opts.getBatch.randSeed);
opts.getBatch.dataPath = opts.dataPath;

%% batch generator
batchWrapper = @(imdb, batch) cnn_getBatch_pairwiseModel(imdb, batch, opts.getBatch) ;

%% training
cnn_train_pairwiseModel(net, imdb, batchWrapper, ...
    opts.train, ...
    'train', find(imdb.images.set == 1), ...
    'val', find(imdb.images.set == 2) );

end












