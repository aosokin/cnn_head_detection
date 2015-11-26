function showFirstLayerFilters( net, varargin )
%showFirstLayerFilters visualizes the filters of the first convolutional layer.
% Input:
%   net - the network in MatConvNet format. Layer for visualization - net.layers{i} with minimum possible i such that isequal( net.layers{iLayer}.type, 'conv')
%   (parName, 

if ~exist('varargin', 'var')
    varargin = {};
end
%% parameters
opts = struct;
opts.filterShowSize = [50 50];
opts.filterRowNumber = 10;
opts = vl_argparse(opts, varargin);

%%
iLayer = 1;
while ~isequal( net.layers{iLayer}.type, 'conv') && iLayer < length(net.layers)
    iLayer = iLayer + 1;
end
if ~isequal( net.layers{iLayer}.type, 'conv')
    error('showFirstLayerFilters:noConv', 'Convolution layer was not found');
end

filters = gather( net.layers{iLayer}.weights{1} );
numFilters = size(filters, 4);
if size(filters, 3) ~= 3
    error('showFirstLayerFilters:badSize', 'Filters have wrong number of channels, only 3 is supported');
end

numRows = opts.filterRowNumber;
numCols = ceil( numFilters / numRows );

filterImage = zeros( numRows * opts.filterShowSize(1), numCols * opts.filterShowSize(2), 3, 'single' );

for iFilter = 1 : numFilters
    iCol = ceil(iFilter / numRows);
    iRow = iFilter - (iCol - 1) * numRows;
    
    
    curX = (1 : opts.filterShowSize(2)) + (iCol - 1) * opts.filterShowSize(2);
    curY = (1 : opts.filterShowSize(1)) + (iRow - 1) * opts.filterShowSize(1);
    
    resizedFilter = imresize(filters(:,:,:,iFilter), opts.filterShowSize, 'nearest');
    filterImage(curY, curX, :) = resizedFilter;
end

maxValue = max(filters(:));
minValue = min(filters(:));

filterImage = (filterImage - minValue) / (maxValue - minValue);

imshow( filterImage );


end

