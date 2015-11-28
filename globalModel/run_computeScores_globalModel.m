%run_computeScores_globalModel applies the global model to the whole input images to produce multi-scale confidence heatmaps

% SETUP THESE PATHS TO RUN THE CODE
dataPath = 'data/HollywoodHeads';
resultPath = 'results/HollywoodHeads';

% network to evaluate
netFile = fullfile( 'models', 'global.mat' );

% file to store the scores
resultFile = fullfile( resultPath, 'global', 'globalModel-scores-test.mat' );

if ~exist(fullfile(resultPath, 'global'), 'dir')
    mkdir(fullfile(resultPath, 'global'));
end

% Choose subset of data to compute the scores.
%   To run the evaluation of the global model you need the test test (3).
scoreSubset = 3; % 1 - train subset, 2 - validation, 3 - test; can do [1,2,3] to compute scores on all the subsets

%% setup paths
filePath = fileparts( mfilename('fullpath') );
run( fullfile( fileparts( filePath ), 'setup.m' ) );

%% parameters
opts_cnn = struct;

opts_cnn.dataPath = dataPath;
opts_cnn.dataset.trainingSetFile = fullfile('Splits', 'train.txt');
opts_cnn.dataset.validationSetFile = fullfile('Splits', 'val.txt');
opts_cnn.dataset.testSetFile = fullfile('Splits', 'test.txt');
opts_cnn.dataset.groundTruthLocalPrefix = 'Annotations';
opts_cnn.dataset.imageLocalPrefix = 'JPEGImages';
opts_cnn.dataset.candidateLocalPrefix = 'Candidates';

opts_cnn.expDir =  resultPath;
opts_cnn.batchSize = 32;
opts_cnn.imdbName = 'globalModel';
opts_cnn.scoreMode = 'beforeSoftMax'; % 'beforeSoftMax' or 'afterSoftMax' or 'scoreDifference';

%% load dataset
opts_cnn.imdbPath = fullfile(opts_cnn.expDir, 'imdb.mat');
generateDataFlag = true;
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


%% select the test set to run evaluation
imageSetToDoPr = false( numel( imdb.images.set ), 1 );
for iSubset = 1 : numel( scoreSubset )
    imageSetToDoPr = imageSetToDoPr | ( imdb.images.set(:) == scoreSubset( iSubset ) );
end
imageSetToDoPr = find( imageSetToDoPr );

%% start the evaluation
opts_cnn.getBatchEvaluation = struct;
opts_cnn.getBatchEvaluation.grid_size = [1 2 4 8];
opts_cnn.getBatchEvaluation.meanImage = net.normalization.averageImage;
opts_cnn.getBatchEvaluation.dataPath = opts_cnn.dataPath;
opts_cnn.getBatchEvaluation.jitterStd = 0;

batchWrapperEvaluation = @(imdb, batch) cnn_getBatch_globalModel(imdb, batch, ...
    opts_cnn.getBatchEvaluation) ;

scores = cnn_computeScores_globalModel( net, imdb, batchWrapperEvaluation, ...
    'imageSet', imageSetToDoPr, ...
    'batchSize', opts_cnn.batchSize, ...
    'scoreMode', opts_cnn.scoreMode);

save( resultFile, 'scores', '-v7.3' );

%% save detections for files
detSavePath = fullfile( resultPath, 'global', 'dets');
if ~exist(detSavePath, 'dir')
    mkdir(detSavePath)
end
detSaveFormat = fullfile(detSavePath, '%s.mat');
disp('Saving detection files');
for i=1:length(imageSetToDoPr)
    imgIdx = imageSetToDoPr(i);
    score = scores{imgIdx};
    %load candidate
    [~, im_name, ~] = fileparts(imdb.imageFiles{imgIdx});
    savePath = sprintf(detSaveFormat, im_name);
    save(savePath, 'score');
end 