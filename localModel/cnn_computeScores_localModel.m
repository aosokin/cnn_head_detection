function [scores, candidateIds] = cnn_computeScores_localModel( net, imdb, getBatch, varargin )
%cnn_computeScores_localModel computes the scores for the images by applying the CNN to all the candidate patches

if ~exist('varargin', 'var')
    varargin = {};
end

%% compute results w.r.t. patches
opts = struct;
opts.gpuBatchSize = 256;
opts.conserveMemory = true;
opts.sync = true;
opts.useGpu = true;
opts.imageSet = 1 : size( imdb.imageFiles, 4 );
opts.scoreMode = 'beforeSoftMax'; % 'beforeSoftMax' or 'afterSoftMax' o 'scoreDifference'

% parse input
opts = vl_argparse(opts, varargin);

%% do the job 
numImages = length( opts.imageSet );
scores = cell( max(opts.imageSet), 1);
candidateIds = cell( max(opts.imageSet), 1);
res = [] ;

for iImageId = 1 : numImages
    iImage = opts.imageSet( iImageId );
    fprintf('Image %d/%d: ', iImageId, numImages);
    tImageStart = tic;
    
    [patchData, patchLabels, patchInfo] = getBatch( imdb, iImage );
    curNumCandidates = size( patchData, 4 );
    fprintf('patches=%d, ', curNumCandidates );

    scores{ iImage } = nan( [curNumCandidates, 1], 'single' );
    candidateIds{ iImage } = int32(patchInfo.candidateIds{1});
    
    numBatches = ceil( curNumCandidates / opts.gpuBatchSize );
    for iBatch = 1 : numBatches
        curIds = (iBatch - 1) * opts.gpuBatchSize + 1 : min( iBatch * opts.gpuBatchSize, curNumCandidates);
        
        im = patchData(:,:,:,curIds);
        labels = ones( size( patchLabels(curIds) ) );
        
        net.layers{end}.class = labels ;
        
        im = gpuArray(im);
        res = vl_simplenn_localModel(net, im, [], res, ...
            'disableDropout', true, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync ) ;
        
        curScores = gather( res(end - 1).x );
        if ~( size(curScores, 1) == 1 && size(curScores, 2) == 1 && size(curScores, 3) == 2 && size(curScores, 4) == length(curIds) )
            error('Scores produced by the network are of strange format');
        end
        curScores = squeeze(curScores);
        
        switch opts.scoreMode
            case 'beforeSoftMax'
                curScores = curScores(1, :);
            case 'afterSoftMax'
                maxVal = max(curScores, [], 1);
                normalizedScores = bsxfun(@minus, curScores, maxVal);
                curScores = normalizedScores(1, :) - log( sum( exp(normalizedScores), 1) );
            case 'scoreDifference'
                curScores = curScores(1, :) - curScores(2, :);
            otherwise
                error('cnn_computeScores_localModel:unknownScoreMode', 'options scoreMode is of incorrect value');
        end
        
        curScores = curScores(:);
        assert( length(curScores) == length(curIds) );
        scores{ iImage }(curIds) = single( curScores );
    end
    fprintf('time: %fs\n', toc(tImageStart) );
end


end

