function scores = cnn_computeScores_globalModel( net, imdb, getBatch, varargin )
%cnn_computeScores_localModel computes the scores for the images by applying the CN to all the candidate patches

if ~exist('varargin', 'var')
    varargin = {};
end

%% compute results w.r.t. patches
opts = struct;
opts.batchSize = 32;
opts.conserveMemory = false;
opts.sync = true;
opts.useGpu = true;
opts.imageSet = 1 : size( imdb.imageFiles, 4 );
opts.scoreMode = 'beforeSoftMax'; % 'beforeSoftMax' or 'afterSoftMax' o 'scoreDifference'
%opts.detSavePathFormat = '';

% parse input
opts = vl_argparse(opts, varargin);

%% do the job 
numImages = length( opts.imageSet );
scores = cell( max(opts.imageSet), 1);

res = [] ;

nBatch = ceil(numImages/opts.batchSize);

for iBatch = 1 : nBatch
    batchid = (iBatch-1)*opts.batchSize+1:min(iBatch*opts.batchSize, numImages);
    batch = opts.imageSet(batchid);
    fprintf('Working with batch %d of %d:\n ', iBatch, nBatch);

    [im, labels] = getBatch( imdb, batch );
    
    tBatchStart = tic;
    im = gpuArray(im);
    
    net.layers{end}.class = labels ;
    res = vl_simplenn_globalModel(net, im, [], res, ...
            'disableDropout', true, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync ) ;
    curScores = gather( res(end - 2).x );
    for j=1:length(batchid)
        score = curScores(:,:,:,j);
        sz = size(score);
        score = squeeze(reshape(score, [sz(1) sz(2) 2 sz(3)/2]));
        scores{opts.imageSet(batchid(j))} = score;
        
        %iImage = opts.imageSet( batchid(j) );
        %[~,f_name,~] = fileparts( imdb.imageFiles{iImage} );
        %save_path = sprintf(opts.detSavePathFormat, f_name);
        %save(save_path, 'score');
    end
    fprintf('forward: %fs\n', toc(tBatchStart) );
end


end

