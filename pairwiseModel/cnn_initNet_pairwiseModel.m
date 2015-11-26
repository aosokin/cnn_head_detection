function net = cnn_initNet_pairwiseModel( networkFile, structuredLoss, numPairwiseClusters, extraLayers, meanImage )
%cnn_initNet_pairwiseModel initializes the pairwise model
% 
% net = cnn_initNet_pairwiseModel( networkFile, structuredLoss, numPairwiseClusters, extraLayers )
%
% Input:
%   networkFile - file with the pretrained network (the dropout layers should already be inside)
%   structuredLoss - type of structure loss to use: 'svmStructCompact' or 'logisticScoresCompact'
%   numPairwiseClusters - number of clusters in the pairwise potentials
%   extraLayers - structure containing information about extra layers (default: [] - no extra layers). Fields:
%       unary - vector containing numbers of nodes in the extra layers for the unary potentials
%       pairwise - vector containing numbers of nodes in the extra layers for the pairwise potentials
%       useDropout - flag showing whether to use dropout in the extra layers
%       cutoffConvolutions - parameter to show if there is need to cut off more layers with weights from the pretrained net
%           One such layer is cut off anyway (connected to the classes from the pretraining task)
%   meanImage - if provided, sets the new mean image for the normalization
%
% Output: 
%   net - structure containing the pairwise model

if ~exist('extraLayers', 'var') || isempty(extraLayers)
    extraLayers = struct;
    extraLayers.unary = [];
    extraLayers.pairwise = [];
    extraLayers.useDropout = true;
    extraLayers.cutoffConvolutions = 0;
end

% the initialization parameters
scal = 1 ;
init_bias = 0.1;
dropout_rate = 0.5;

%% read the network file
pretrainedNet = load( networkFile, '-mat' );
if ~isfield(pretrainedNet, 'layers')
    if ~isfield(pretrainedNet, 'net')
        error('cnn_initNet_pairwiseModel:incorrectInitialization', 'File to initialize the network is of incorrect format')
    else
        pretrainedNet = pretrainedNet.net;
    end
end
pretrainedNet.layers(end) = []; % cut off the loss layer
if ~exist('meanImage', 'var')
    meanImage = pretrainedNet.normalization.averageImage;
end

%% determine the number of output features
% find the last convolutional layer
lastConvolutionalLayer = length(pretrainedNet.layers);
cutoffConvolutions = 0;
while (~isequal( pretrainedNet.layers{ lastConvolutionalLayer }.type, 'conv' ) || cutoffConvolutions < extraLayers.cutoffConvolutions) ...
        && lastConvolutionalLayer > 1 
    if isequal( pretrainedNet.layers{ lastConvolutionalLayer }.type, 'conv' )
        cutoffConvolutions = cutoffConvolutions + 1;
    end
    lastConvolutionalLayer = lastConvolutionalLayer - 1;
end
if isequal( pretrainedNet.layers{ lastConvolutionalLayer }.type, 'conv' )
    numFeatures = size( pretrainedNet.layers{ lastConvolutionalLayer }.weights{1}, 3 );
else
    error('cnn_initNet_pairwiseModel:wrongNetwork', 'Cannot determine the number of features from the pretrained network');
end

%% create the feature extractor network
net = struct;
net.featureExtractor = struct;
net.featureExtractor.layers = pretrainedNet.layers;
net.featureExtractor.layers(lastConvolutionalLayer : 1 : end) = []; % cut off the last layer

%% add the loss layer
net.lossLayer = struct;
net.lossLayer.type = structuredLoss;

numClasses = 2;
switch net.lossLayer.type
    case 'svmStructCompact'
        numUnaryOutputs = 1;
        numPairwiseOutputs = numPairwiseClusters;
    case 'logisticScoresCompact'
        numUnaryOutputs = 1;
        numPairwiseOutputs = numPairwiseClusters;
    otherwise
        error('cnn_initNet_pairwiseModel:unknownLoss',['The structured loss is not recognised: ', net.lossLayer.type]);
end

%% add the unary network
net.unaryNetwork = struct;
net.unaryNetwork.layers = cell(0, 0);

numExtraLayers = length(extraLayers.unary);
numNodesExtraLayersUnary = [ numFeatures; extraLayers.unary(:)];
for iLayer = 1 : numExtraLayers
    net.unaryNetwork.layers{end+1} = struct('type', 'conv', ...
        'weights', {{ 0.01/scal * randn(1, 1, numNodesExtraLayersUnary(iLayer), numNodesExtraLayersUnary(iLayer+1),'single'),... % filters
        init_bias*ones(1,numNodesExtraLayersUnary(iLayer+1),'single') }}, ... % biases
        'stride', 1, ...
        'pad', 0, ...
        'learningRate', [1, 2], ...
        'weightDecay', [1, 0], ...
        'name', ['unaryNetwork_conv', num2str(iLayer)]) ;
    net.unaryNetwork.layers{end+1} = struct('type', 'relu') ;
    if extraLayers.useDropout
        net.unaryNetwork.layers{end+1} = struct('type', 'dropout', 'rate', dropout_rate) ;
    end
end
net.unaryNetwork.layers{end+1} = struct('type', 'conv', ...
    'weights', {{ 0.01/scal * randn(1, 1, numNodesExtraLayersUnary(end), numUnaryOutputs,'single'),... % weights
    init_bias * ones(1, numUnaryOutputs, 'single') }}, ... % biases
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1, 2], ...
    'weightDecay', [1, 0], ...
    'name', 'unaryNetwork_convLast') ;

%% add the pairwise network
net.pairwiseNetwork = struct;
net.pairwiseNetwork.layers = cell(0, 0);

numExtraLayers = length(extraLayers.pairwise);
numNodesExtraLayersPairwise = [ 2 * numFeatures; extraLayers.pairwise(:)];
for iLayer = 1 : numExtraLayers
    net.pairwiseNetwork.layers{end+1} = struct('type', 'conv', ...
        'weights', {{ 0.01/scal * randn(1, 1, numNodesExtraLayersPairwise(iLayer), numNodesExtraLayersPairwise(iLayer+1),'single'), ... % filters
        init_bias*ones(1,numNodesExtraLayersPairwise(iLayer+1),'single') }}, ... % biases
        'stride', 1, ...
        'pad', 0, ...
        'learningRate', [1, 2], ...
        'weightDecay', [1, 0], ...
        'name', ['pairwiseNetwork_conv', num2str(iLayer)]) ;
    net.pairwiseNetwork.layers{end+1} = struct('type', 'relu') ;
    if extraLayers.useDropout
        net.pairwiseNetwork.layers{end+1} = struct('type', 'dropout', 'rate', dropout_rate) ;
    end
end
net.pairwiseNetwork.layers{end+1} = struct('type', 'conv', ...
    'weights', {{ 0.01/scal * randn(1, 1, numNodesExtraLayersPairwise(end), numPairwiseOutputs,'single'), ... % filters
    init_bias * ones(1, numPairwiseOutputs, 'single') }}, ... % biases
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1, 2], ...
    'weightDecay', [1, 0], ...
    'name', 'pairwiseNetwork_convLast') ;

%% collect all trainable layers
net.layers = cell(0,0);

for iLayer = 1 : length(net.featureExtractor.layers)
    if isequal( net.featureExtractor.layers{iLayer}.type, 'conv' )
        net.layers{end+1} = net.featureExtractor.layers{iLayer};
        net.featureExtractor.layers{iLayer} = struct( 'type', 'convPtr', 'index', length(net.layers) );
    end
end
for iLayer = 1 : length(net.unaryNetwork.layers)
    if isequal( net.unaryNetwork.layers{iLayer}.type, 'conv' )
        net.layers{end+1} = net.unaryNetwork.layers{iLayer};
        net.unaryNetwork.layers{iLayer} = struct( 'type', 'convPtr', 'index', length(net.layers) );
    end
end
for iLayer = 1 : length(net.pairwiseNetwork.layers)
    if isequal( net.pairwiseNetwork.layers{iLayer}.type, 'conv' )
        net.layers{end+1} = net.pairwiseNetwork.layers{iLayer};
        net.pairwiseNetwork.layers{iLayer} = struct( 'type', 'convPtr', 'index', length(net.layers) );
    end
end

%% mean image normalization
% set the new mean image
net.normalization.averageImage = meanImage;

%% class order
net.classes = struct;
net.classes.name = {'head', 'background'};
net.classes.description = {'head', 'background'};

end
