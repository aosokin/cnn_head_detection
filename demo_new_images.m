% Demo code
% Tuan-Hung Vu, Anton Osokin, Ivan Laptev, Context-aware CNNs for person head detection, ICCV 2015
% This script shows how to try our local model on some new images.

% SETUP THESE PATHS TO RUN THE CODE
dataPath = 'data/new_data';
resultPath = 'results/new_data';

% Put the images into 'data/new_data/images'
% To get an example image run
% wget -P data/new_data/images http://tech.velmont.net/files/2009/04/lenna-lg.jpg

%% Setup
matconvnetPath = '~/local/software/matlab_toolboxes/matconvnet-1.0-beta18';
setup( matconvnetPath );
 
% cudaRoot = '/usr/cuda-7.0' ;
% compile_mex(cudaRoot);

% Assumes that Selective search is downloaded like this:
% wget http://huppelen.nl/publications/SelectiveSearchCodeIJCV.zip
% unzip SelectiveSearchCodeIJCV.zip
addpath('SelectiveSearchCodeIJCV', 'SelectiveSearchCodeIJCV/Dependencies');


if ~exist(fullfile(dataPath, 'splits'), 'dir')
    mkdir(fullfile(dataPath, 'splits'));
end
if ~exist(fullfile(dataPath, 'candidates'), 'dir')
    mkdir(fullfile(dataPath, 'candidates'));
end

all_files = dir(fullfile(dataPath, 'images'));
image_names = cell(0,0);
for i_file = 1 : length(all_files)
    if length(all_files(i_file).name) > 2
        [~, cur_image_name, ~] = fileparts(all_files(i_file).name);
        image_names{end+1} = cur_image_name;
    end
end
writeLines( fullfile(dataPath, 'splits', 'test.txt'), image_names );

% network to evaluate
netFile = fullfile( 'models', 'local.mat' );

% file to store the scores
resultFile = fullfile( resultPath, 'local', 'localModel-scores-test.mat' );

% Only test images are used
scoreSubset = 3; 

%% Compute candidates by using Selective Search on all the images
image_files = dir(fullfile(dataPath, 'images','*.jpg'));
fprintf('Running Selective Search on %d images\n', length(image_files));
for i_image = 1 : length(image_files)
    fprintf('Image %d of %d\n', i_image, length(image_files));
    curImage = imread(fullfile(dataPath, 'images', image_files(i_image).name));
    boxes = selective_search_boxes(curImage, true);
    [~,image_name,~] = fileparts(image_files(i_image).name);
    save( fullfile(dataPath, 'candidates', [image_name, '.mat']), 'boxes')
end

%% parameters
opts_cnn = struct;
opts_cnn.dataPath = dataPath;
opts_cnn.dataset.testSetFile = fullfile('splits', 'test.txt');
opts_cnn.dataset.imageLocalPrefix = 'images';
opts_cnn.dataset.candidateLocalPrefix = 'candidates';
opts_cnn.expDir =  resultPath;
opts_cnn.scoreMode = 'scoreDifference'; % 'beforeSoftMax' or 'afterSoftMax' or 'scoreDifference';

%% load dataset
opts_cnn.imdbPath = fullfile(opts_cnn.expDir, 'imdb.mat');
fprintf('Generating imdb file %s\n', opts_cnn.imdbPath);
imdb = cnn_prepareData_localModel( opts_cnn.dataset, 'dataPath', opts_cnn.dataPath );
imdb.opts = opts_cnn.dataset;
if ~exist(opts_cnn.expDir, 'dir')
    mkdir(opts_cnn.expDir);
end
save(opts_cnn.imdbPath, '-struct', 'imdb', '-v7.3') ;

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
opts_cnn.getBatchEvaluation.maxGpuImages = 128;
opts_cnn.getBatchEvaluation.dataPath = opts_cnn.dataPath;
opts_cnn.getBatchEvaluation.jitterStd = 0;
opts_cnn.getBatchEvaluation.numPatchesPerImage = inf;

batchWrapperEvaluation = @(imdb, batch) cnn_getBatch_localModel(imdb, batch, ...
    opts_cnn.getBatchEvaluation) ;

[scores, candidateIds] = cnn_computeScores_localModel( net, imdb, batchWrapperEvaluation, ...
    'imageSet', imageSetToDoPr, ...
    'gpuBatchSize', opts_cnn.getBatchEvaluation.maxGpuImages, ...
    'scoreMode', opts_cnn.scoreMode );

if exist(resultFile, 'file')
    warning('The results file already exists. Overwriting!');
end
if ~exist(fileparts(resultFile), 'dir')
    mkdir(fileparts(resultFile));
end
save( resultFile, 'scores', 'candidateIds', '-v7.3' );

%% vizualize the detections
iImage = 1;
curImage = imread( fullfile(opts_cnn.dataPath, imdb.imageFiles{iImage}) );
curCandidates = load( fullfile(opts_cnn.dataPath, imdb.candidateFiles{iImage}), 'boxes' );
curCandidates = convertBb_Y1X1Y2X2_to_X1Y1WH(curCandidates.boxes);

idsNms = selectBoundingBoxesNonMaxSup( curCandidates(candidateIds{iImage},:), scores{iImage});

toPlotCandidates = curCandidates( candidateIds{iImage}(idsNms(1)), :);

imageWithBoxes = showBoundingBoxes(curImage, toPlotCandidates, 'y');
imshow(imageWithBoxes);
