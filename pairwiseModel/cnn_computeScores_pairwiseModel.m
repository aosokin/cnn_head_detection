function [scores, candidateIds] = cnn_computeScores_pairwiseModel( net, imdb, getBatch, varargin )
%cnn_computeScores_pairwiseModel computes the scores using the structure network

if ~exist('varargin', 'var')
    varargin = {};
end

%% compute results w.r.t. patches
opts = struct;
opts.conserveMemory = true;
opts.sync = true;
opts.imageSet = 1 : size( imdb.imageFiles, 4 );
opts.scoreMode = 'maxMarginals';
% parse input
opts = vl_argparse(opts, varargin);

%% do the job
numImages = length( opts.imageSet );
scores = cell( max(opts.imageSet), 1);
candidateIds = cell( max(opts.imageSet), 1);

for iImageId = 1 : numImages
    iImage = opts.imageSet( iImageId );
    fprintf('Image %d/%d: ', iImageId, numImages);
    tImageStart = tic;
    
    [patchData, patchLabels, patchInfo] = getBatch( imdb, iImage );
    curNumCandidates = size( patchData, 4 );
    fprintf('patches=%d, ', curNumCandidates );
    
    [~, ~, predictions] = vl_structuredNetwork_pairwiseModel(net, patchData, [], patchLabels, [], ...
        'computeMaxMarginals', true, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);
    maxMarginals = predictions{1}.maxMarginals;
    
    candidateIds{ iImage } = int32( patchInfo.candidateIds{1}(:) );
    
    
    switch opts.scoreMode
        case 'maxMarginals'
            curScores = maxMarginals(:, 1) - maxMarginals(:, 2);
        otherwise
            error('cnn_computeScores_pairwiseModel:unknownScoreMode', 'options scoreMode is of incorrect value');
    end
    
    curScores = curScores(:);
    if numel(curScores) ~= numel( candidateIds{iImage} )
        error('cnn_computeScores_pairwiseModel:inconsistentCandidates', 'something went wrong')
    end
    scores{ iImage } = single( curScores );
    
    fprintf('time: %fs\n', toc(tImageStart) );
end


end

