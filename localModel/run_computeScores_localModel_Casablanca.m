%run_computeScores_localModel applies the local model to the bounding-box proposals to compute their scores

% SETUP THESE PATHS TO RUN THE CODE
dataPath = 'data/Casablanca';
resultPath = 'results/Casablanca';

% network to evaluate
netFile = fullfile( 'models', 'local.mat' );

% file to store the scores
resultFile = fullfile( resultPath, 'local', 'localModel-scores-test.mat' );

% Casablanca dataset contains only the test set, so scoreSubset has to be equal to 3
scoreSubset = 3; 

%% setup paths
filePath = fileparts( mfilename('fullpath') );
run( fullfile( fileparts( filePath ), 'setup.m' ) );

%% parameters
opts_cnn = struct;
opts_cnn.dataPath = dataPath;
opts_cnn.dataset.testSetFile = fullfile('Splits', 'test.txt');
opts_cnn.dataset.groundTruthLocalPrefix = 'Annotations';
opts_cnn.dataset.imageLocalPrefix = 'JPEGImages';
opts_cnn.dataset.candidateLocalPrefix = 'Candidates';

opts_cnn.expDir =  resultPath;
opts_cnn.maxGpuImagesEvaluation = 3000;
opts_cnn.gpuBatchSize = 128;
opts_cnn.scoreMode = 'beforeSoftMax'; % 'beforeSoftMax' or 'afterSoftMax' or 'scoreDifference';

opts_cnn.evaluation.iouThreshold = 0.5;
opts_cnn.evaluation.nmsMaxCandidateNumber = inf;
opts_cnn.evaluation.useDifficultImages = false;

%% load dataset
opts_cnn.imdbPath = fullfile(opts_cnn.expDir, 'imdb.mat');
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
    imdb = cnn_prepareData_localModel( opts_cnn.dataset, 'dataPath', opts_cnn.dataPath );
    imdb.opts = opts_cnn.dataset;
    mkdir(opts_cnn.expDir);
    save(opts_cnn.imdbPath, '-struct', 'imdb', '-v7.3') ;
end

%% load network
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
opts_cnn.getBatchEvaluation.cropMode = 'warp';
opts_cnn.getBatchEvaluation.cropPad = [18, 18, 18, 18];
opts_cnn.getBatchEvaluation.meanImage = net.normalization.averageImage;
opts_cnn.getBatchEvaluation.numPatchesPerImage = inf;
opts_cnn.getBatchEvaluation.maxPositives = inf;
opts_cnn.getBatchEvaluation.maxGpuImages = opts_cnn.maxGpuImagesEvaluation;
opts_cnn.getBatchEvaluation.iouPositiveThreshold = opts_cnn.evaluation.iouThreshold;
opts_cnn.getBatchEvaluation.iouNegativeThreshold = opts_cnn.evaluation.iouThreshold;
opts_cnn.getBatchEvaluation.dataPath = opts_cnn.dataPath;
opts_cnn.getBatchEvaluation.jitterStd = 0;

batchWrapperEvaluation = @(imdb, batch) cnn_getBatch_localModel(imdb, batch, ...
    opts_cnn.getBatchEvaluation) ;

[scores, candidateIds] = cnn_computeScores_localModel( net, imdb, batchWrapperEvaluation, ...
    'imageSet', imageSetToDoPr, ...
    'gpuBatchSize', opts_cnn.gpuBatchSize, ...
    'scoreMode', opts_cnn.scoreMode );

if exist(resultFile, 'file')
    warning('The results file already exists. Overwriting!');
end
if ~exist(fileparts(resultFile), 'dir')
    mkdir(fileparts(resultFile));
end
save( resultFile, 'scores', 'candidateIds', '-v7.3' );

%% save detections to files
detSavePath = fullfile( resultPath, 'local', 'dets');
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
