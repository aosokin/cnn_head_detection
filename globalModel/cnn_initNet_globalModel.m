function net = cnn_initNet_globalModel( networkFile, numOutput, numClasses, meanImage, numNodesExtraLayers, addDropout )
%cnn_initNet_localModel initializes CNN from a pretrained network in MatConvNet format
%
% net = cnn_initNet_globalModel( networkFile, numOutput, numClasses);
% net = cnn_initNet_globalModel( networkFile, numOutput, numClasses, meanImage );
% net = cnn_initNet_globalModel( networkFile, numOutput, numClasses, meanImage, numNodesExtraLayers );
% net = cnn_initNet_globalModel( networkFile, numOutput, numClasses, meanImage, numNodesExtraLayers, addDropout );
%
% Input: 
%   networkFile - file with the pretrained network
%   numOutput - total number of heatmap cells across all scales
%   numClasses - number of class for which to construct the network
%   meanImage - the new mean image of the network
%   numNodesExtraLayers - number of nodes to add to the extra layers, weights are initialized randomly (default: [])
%   addDropout - true of false whether to add dropout layers after the fully-connected layers (default: false)
%
% Output: 
%   net - struct containing the network in the matconvnet format

if ~exist('numNodesExtraLayers', 'var') || isempty(numNodesExtraLayers)
    numNodesExtraLayers = [];
end
if ~exist('addDropout', 'var') || isempty(addDropout)
    addDropout = false;
end
if ~exist('meanImage', 'var') || isempty(meanImage)
    meanImage = [];
end


% the initialization parameters
scal = 1 ;
init_bias = 0.1;
dropout_rate = 0.5;

%% read the network file
net = load( networkFile );
net.layers(end - 1 : end) = []; % cut off the loss layer

%% determine the number of output features
% find the last convolutional layer
iLayer = length(net.layers);
while ~isequal( net.layers{ iLayer }.type, 'conv' ) && iLayer > 1
    iLayer = iLayer - 1;
end
if isequal( net.layers{ iLayer }.type, 'conv' )
    numFeatures = size( net.layers{ iLayer }.weights{1}, 4 );
else
    error('cnn_initNet_localModel:wrongNetwork', 'Cannot determine the number of features from the pretrained network');
end

%% regularize the fully connected layers by dropout
if addDropout
    numDropout = 0;
    iLayer = 1;
    while iLayer <= length(net.layers)
        if isequal( net.layers{iLayer}.type, 'conv' ) && isequal( net.layers{iLayer}.name(1 : 2), 'fc' )
            % if iLayer is the fully connected layer insert the dropout layer
            curNumLayers = length(net.layers);
            if iLayer < curNumLayers % if this is not the last layer
                % move all the remaining layers one layer forward
                net.layers(iLayer + 2 : curNumLayers + 1) = net.layers(iLayer + 1 : curNumLayers);
            end
            numDropout = numDropout + 1;
            net.layers{iLayer+1} = struct('type', 'dropout', ...
                'rate', dropout_rate, ...
                'name', ['dropout', num2str(numDropout)] );
        end
        iLayer = iLayer + 1;
    end
end

%% add new layers
numExtraLayers = length(numNodesExtraLayers);
numNodesExtraLayers = [numFeatures; numNodesExtraLayers(:)];
for iLayer = 1 : numExtraLayers
    net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{ 0.01/scal * randn(1, 1, numNodesExtraLayers(iLayer), numNodesExtraLayers(iLayer+1),'single'),... % filters
          init_bias*ones(1,numNodesExtraLayers(iLayer+1),'single') }} , ... % biases
        'stride', 1, ...
        'pad', 0, ...
        'learningRate', [1, 2], ...
        'weightDecay', [1, 0], ...
        'name', ['fc_extra', num2str(iLayer)]) ;
    net.layers{end+1} = struct('type', 'dropout', ...
                               'rate', dropout_rate, ...
                               'name', ['dropout_extra', num2str(iLayer)] );
    net.layers{end+1} = struct('type', 'relu', ...
                               'name', ['relu_extra', num2str(iLayer)]) ;
end
net.layers{end+1} = struct('type', 'conv', ...
     'weights', {{ 0.01/scal * randn(1, 1, numNodesExtraLayers(end), numOutput*numClasses,'single'),... % filters
       init_bias*ones(1,numOutput*numClasses,'single') }}, ... % biases
     'stride', 1, ...
     'pad', 0, ...
     'learningRate', [1, 2], ...
     'weightDecay', [1, 0], ...
     'name', 'fc_classes') ;

% reshape layer
net.layers{end+1} = struct('type', 'reshape',...
    'numOutput', numOutput,...
    'numClasses', numClasses);
% The loss
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.classes = struct;
net.classes.name = {'head', 'background'};
net.classes.description = {'head', 'background'};

% set the new mean image
if ~isempty(meanImage)
    net.normalization = struct;
    net.normalization.averageImage = meanImage;
end

end
