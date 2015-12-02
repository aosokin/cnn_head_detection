function [im, labels, patchInfo] = cnn_getBatch_globalModel(imdb, batch, varargin)
%cnn_getBatch_localModel constructs the batch to train the local model

tVeryStart = tic;

%% parse parameters
opts = struct;
opts.meanImage = nan;
opts.numThreads = 4;
opts.maxGpuImages = 256;
opts.grid_size = [1 2 4 8];
opts.nstride = 2;
opts.ov_thres = 0.3;

opts.jitterStd = 0;
opts.dataPath = '';
opts.jpegImagesFlag = true;
opts = vl_argparse(opts, varargin);

if isnan(opts.meanImage)
    error('cnn_getBatch_globalModel:noMeanImage', 'mean image parameter is not specified');
end

if (size(opts.meanImage, 1) ~= size(opts.meanImage, 2))
    error('cnn_getBatch_globalModel:sizeMeanImage', 'Currently only square input is supported');
end

opts.afterCropSize = [ size(opts.meanImage, 1), size(opts.meanImage, 2) ];

tStart = tic;

%% extract data
numImages = length(batch);
batchGt = cell(numImages, 1);
batchImages = cell(numImages, 1);

% get full paths for the batch images
jpegImagesFlag = opts.jpegImagesFlag;
curImageNames = cell( numel(batch), 1 );
for iImage = 1 : numel(batch)
    curImageNames{iImage} = fullfile( opts.dataPath, imdb.imageFiles{ batch(iImage) } );

    % check if all the images are in JPEG format
    [~,~,curExt] = fileparts( curImageNames{iImage} );
    if ~isequal( curExt, '.jpeg') && ~isequal( curExt, '.jpg')
        jpegImagesFlag = false;
    end
end

% read the images
if jpegImagesFlag
    batchImages_ori = vl_imreadjpeg( curImageNames, 'NumThreads', opts.numThreads);
else
    for iImage = 1 : numImages
        batchImages_ori{iImage} = single( imread( curImageNames{iImage} ) );
    end
end

for iImage = 1 : numImages
    pad_size(iImage) = floor((size(batchImages_ori{iImage},2)-size(batchImages_ori{iImage},1))/2);
    im_pad = padarray(batchImages_ori{iImage}, [pad_size(iImage) 0]);
    batchImages{iImage} = single(imresize(im_pad, opts.afterCropSize));
    scale_factor(iImage) = opts.afterCropSize(1)/size(im_pad,1);
end


% read other data
input_size = size(opts.meanImage, 1);
for n_tile=opts.grid_size
    tile_size = input_size/n_tile;
    stride = tile_size/opts.nstride;
    n_tile_strided{n_tile} = floor((input_size-stride)/stride);
end

for iImage = 1 : numImages
    GT = imdb.groundTruth{ batch(iImage) };

    GT_pad = GT;
    GT_pad(:, [2 4]) = GT_pad(:, [2 4]) + pad_size(iImage);
    GT_pad = GT_pad*scale_factor(iImage);
    
    for n_tile=opts.grid_size
        hm{n_tile} = 2*ones(n_tile_strided{n_tile});
    end

    for GT_cnt= 1:size(GT_pad, 1)
        max_ov_allscale = -inf;
        max_c_allscale = 0;
        max_d_allscale = 0;
        max_n_tile = 0;
        has_ov_cell = false;
        
        for n_tile=opts.grid_size
            tile_size = input_size/n_tile;
            stride = tile_size/opts.nstride;
            
            %determine which cell the top-left belonging to
            cell_c_tl = floor((GT_pad(GT_cnt, 1)-1)/stride);
            cell_d_tl = floor((GT_pad(GT_cnt, 2)-1)/stride);
            
            %determine which cell the bot_right belonging to
            cell_c_br = floor((GT_pad(GT_cnt, 3)-1)/stride);
            cell_d_br = floor((GT_pad(GT_cnt, 4)-1)/stride);

            cell_c_tl = min(max(1, cell_c_tl), n_tile_strided{n_tile});
            cell_d_tl = min(max(1, cell_d_tl), n_tile_strided{n_tile});
            cell_c_br = min(max(1, cell_c_br), n_tile_strided{n_tile});
            cell_d_br = min(max(1, cell_d_br), n_tile_strided{n_tile});

            max_ov = -inf;
            max_c = 0;
            max_d = 0;
            for c = cell_c_tl:cell_c_br
                for d = cell_d_tl:cell_d_br
                    ov = bbIntersectionOverUnion([(c-1)*stride+1 (d-1)*stride+1 tile_size tile_size], convertBb_X1Y1X2Y2_to_X1Y1WH(GT_pad(GT_cnt, 1:4)));

                    if (ov> opts.ov_thres)
                        hm{n_tile}(d,c) = 1;
                        has_ov_cell = true;
                    end

                    if (ov > max_ov)
                        max_ov = ov;
                        max_c = c;
                        max_d = d;
                    end                 
                 end
            end

            if (max_ov > max_ov_allscale)
                 max_ov_allscale = max_ov;
                 max_n_tile = n_tile;
                 max_c_allscale = max_c;
                 max_d_allscale = max_d;
            end
       end
       if (~has_ov_cell)
           if (max_ov_allscale > 0)
               hm{max_n_tile}(max_d_allscale, max_c_allscale) = 1;
           end
       end
   end


   output = [];
   cnt = 0;
   for n_tile=opts.grid_size
       output(cnt+1:cnt+n_tile_strided{n_tile}^2) = reshape(hm{n_tile}, 1, n_tile_strided{n_tile}^2);
       cnt = cnt+n_tile_strided{n_tile}^2;
   end

   batchGt{iImage} = output; %start indexing from 1 + reverse index
end
readTime = toc(tStart);

%% create batch
im = zeros( size(opts.meanImage, 1), size(opts.meanImage, 2), 3, numImages, 'single', 'gpuArray' );
labels = [];

for iImage = 1 : numImages
    curImage = single( batchImages{iImage} );
    
    % form patches
    if ~isempty(im)
        im(:, :, :, iImage) = curImage;
        labels(end+1:end+length(batchGt{iImage})) = batchGt{iImage};
    else
        im = curImage;
        labels = batchGt{iImage} ;
    end
end

im = bsxfun(@minus, im, opts.meanImage);
labels = labels(:);

patchInfo = struct;
patchInfo.readTime = readTime;
patchInfo.batchTime = toc(tVeryStart);
patchInfo.wasteTime = patchInfo.batchTime - readTime;

end
