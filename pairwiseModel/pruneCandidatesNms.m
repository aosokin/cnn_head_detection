function [ newScores, newCandidateIds ] = pruneCandidatesNms( imdb, scoreFile, varargin )
%pruneCandidatesNms is a part of cnn_prepareData_pairwiseModel.m which preprocesses for the training of the pairwise model

if ~exist('varargin', 'var')
    varargin = {};
end
%% parse parameters
opts = struct;
opts.maxNumPatchesPerImage = 16;
opts.nmsIntersectionOverAreaThreshold = 0.3;
opts.numThreads = 4;
opts.dataPath = '';
% parse input
opts = vl_argparse(opts, varargin);

if ~exist('scoreFile', 'var') || isempty(scoreFile) || ~exist( fullfile(scoreFile), 'file')
    if ~isempty(opts.dataPath)
        scoreFile = fullfile(opts.dataPath, scoreFile);
        if ~exist( fullfile(scoreFile), 'file')
            error('pruneCandidatesNms:noScoreFile', 'Cannot get the file with the precomputed scores');
        end
    end
end

%% do the job
fprintf('Reading precomputed scores ... ');
tStart = tic;
load( scoreFile, 'scores', 'candidateIds' );
fprintf( '%fs\n', toc(tStart) );

fprintf('Pruning candidates with NMS ... ');
tStart = tic;
numImages = length( imdb.imageFiles );
newScores = cell(numImages, 1);
newCandidateIds = cell(numImages, 1);
candidateFiles = imdb.candidateFiles;
curPrefix = opts.dataPath;
parfor (iImage = 1 : numImages, opts.numThreads)
% for iImage = 1 : numImages
    if isempty(scores{iImage}) || isempty(candidateIds{iImage})
        continue;
    end
    if mod(iImage, 1000) == 0
        fprintf('Image %d\n', iImage);
    end
    
    curCandidates = load( fullfile(curPrefix, candidateFiles{ iImage } ) );
    
    curCandidates = curCandidates.boxes( candidateIds{ iImage }, :);
    curScores = scores{ iImage };
    
    % fix the Bb format: SelectiveSearch format [y1 x1 y2 x2] to format [x y w h]
    curBb = convertBb_Y1X1Y2X2_to_X1Y1WH( curCandidates(:, 1 : 4) );
    
    %select only BBs with high scores and non-max-sup
    idsNms = selectBoundingBoxesNonMaxSup( curBb, curScores, ...
        'numBoundingBoxMax', opts.maxNumPatchesPerImage, ...
        'nmsIntersectionOverAreaThreshold', opts.nmsIntersectionOverAreaThreshold);
    
    newCandidateIds{iImage} = candidateIds{ iImage }(idsNms);
    newScores{iImage} = curScores(idsNms);
end
fprintf( '%fs\n', toc(tStart) );

end

