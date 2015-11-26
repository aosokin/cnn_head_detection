function [lossValue, gradients, predictions] = vl_structuredNetwork_pairwiseModel(net, x, gradients, labels, dzdy, varargin)
%vl_structuredNetwork_pairwiseModel implements the evaluation of the structured network together with its gradient comuptation

opts = struct;
opts.conserveMemory = true ;
opts.sync = true ;
opts.disableDropout = false ;
opts.backPropagateType = 'all'; % 'all', 'unaryAndPairwise', 'onlyUnary', 'onlyPairwise'
opts.computeMaxMarginals = false;
opts.disableDropoutFeatureExtractor = false;
opts.lossNormalization = 1;
opts = vl_argparse(opts, varargin);

if (nargin <= 4) || isempty(dzdy)
    doder = false ;
else
    doder = true ;
end

%% preparation
numImages = length(labels);
numNodes = zeros(numImages, 1);
numEdges = zeros(numImages, 1);
for iImage = 1 : numImages
    numNodes( iImage ) = length( labels{iImage}.candidateBatchIds );
    numEdges( iImage ) = size(labels{ iImage }.clusteredEdges.bbIds, 1);
end
numCandidates = sum(numNodes);
if size(x, 4) ~= numCandidates
    error('vl_structuredNetwork_pairwiseModel:badSizeInput', 'Size of data input "x" is incompatible with "labels"');
end

n = numel(net.layers) ;
if doder
    if isempty(gradients)
        gradients = cell( length(net.layers), 1);
    end
    for i = 1 : n
        if isfield( net.layers{i}, 'weights' )
            for j = 1 : length( net.layers{i}.weights )
                gradients{i}.dzdw{j} = zeros( size(net.layers{i}.weights{j}), 'like', net.layers{i}.weights{j} );
            end
        end
    end
end

%% forward pass
% forward pass on the feature extraction network
disableFeatureExtractorDropout = opts.disableDropout || opts.disableDropoutFeatureExtractor;
numLayers_features = length(net.featureExtractor.layers);
res_features = initRes( numLayers_features );
res_features = vl_simplenn_pairwiseModel_forwardPass(net.featureExtractor, x, net.layers, res_features, ...
    'conserveMemory', opts.conserveMemory, ...
    'sync', opts.sync, ...
    'disableDropout', disableFeatureExtractorDropout);

extractedFeatures = res_features(end).x;
numFeatures = size( extractedFeatures, 3 );

% forward pass on the unary potential network
numLayers_unary = length(net.unaryNetwork.layers);
res_unary = initRes( numLayers_unary );
res_unary = vl_simplenn_pairwiseModel_forwardPass( net.unaryNetwork, extractedFeatures, net.layers, res_unary, ...
    'conserveMemory', opts.conserveMemory, ...
    'sync', opts.sync, ...
    'disableDropout', opts.disableDropout);

% forward pass on the pairwise potential network
x_pairwise = zeros( 1, 1, numFeatures * 2, sum(numEdges), 'like', x);
startId = 0;
batchId_inEdge = zeros( sum(numEdges), 2 );
for iImage = 1 : numImages
    curIds = startId + (1 : numEdges(iImage));
    curCandidateBatchIds = labels{ iImage }.candidateBatchIds( labels{ iImage }.clusteredEdges.bbIds );
    x_pairwise(:,:,1:numFeatures,curIds) = extractedFeatures(:,:,:, curCandidateBatchIds(:,1) );
    x_pairwise(:,:,numFeatures + 1 : 2 * numFeatures,curIds) = extractedFeatures(:,:,:, curCandidateBatchIds(:,2) );
    
    batchId_inEdge( curIds, :) = curCandidateBatchIds;
    
    startId = startId + numEdges(iImage);
end

numLayers_pairwise = length(net.pairwiseNetwork.layers);
res_pairwise = initRes( numLayers_pairwise );
res_pairwise = vl_simplenn_pairwiseModel_forwardPass( net.pairwiseNetwork, x_pairwise, net.layers, res_pairwise, ...
    'conserveMemory', opts.conserveMemory, ...
    'sync', opts.sync, ...
    'disableDropout', opts.disableDropout);

%% loss evaluation
unaryPotentials = gather(res_unary(end).x);
pairwisePotentials = gather(res_pairwise(end).x);

switch net.lossLayer.type
    case 'logisticScoresCompact'
        [lossValue, unaryDerivative, pairwiseDerivative, predictions] = vl_logisticScoreLoss_pairwiseCompactModel( unaryPotentials, pairwisePotentials, labels, dzdy, opts.computeMaxMarginals);
    case 'svmStructCompact'
        [lossValue, unaryDerivative, pairwiseDerivative, predictions] = vl_svmStructLoss_pairwiseCompactModel( unaryPotentials, pairwisePotentials, labels, dzdy, opts.computeMaxMarginals, opts.lossNormalization);
    otherwise
        error( ['Unknown structured loss: ', net.lossLayer.type] );
end

%% backward pass
if doder
    if isequal( opts.backPropagateType, 'all' ) || isequal( opts.backPropagateType, 'onlyUnary' ) || isequal( opts.backPropagateType, 'unaryAndPairwise' )
        % backward pass on the unary potential network
        [res_unary, gradients] = vl_simplenn_pairwiseModel_backwardPass( net.unaryNetwork, extractedFeatures, net.layers, res_unary, gradients, unaryDerivative, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync, ...
            'disableDropout', opts.disableDropout );
    end
    
    if isequal( opts.backPropagateType, 'all' )
        % backward pass of the unary signal through the feature extracting network
        % start accumulating data to backprop through the feature extraction network:
        featureBackPropSignal = res_unary(1).dzdx;
    end
    
    if isequal( opts.backPropagateType, 'all' ) || isequal( opts.backPropagateType, 'onlyPairwise' ) || isequal( opts.backPropagateType, 'unaryAndPairwise' )
        % backward pass through pairwise potentials network
        [res_pairwise, gradients] = vl_simplenn_pairwiseModel_backwardPass( net.pairwiseNetwork, x_pairwise, net.layers, res_pairwise, gradients, pairwiseDerivative, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync, ...
            'disableDropout', opts.disableDropout );
    end
    
    % prepare data for passing backward pairwise signal
    if isequal( opts.backPropagateType, 'all' )
        I = eye( numCandidates, 'like', x );
        ids1 = I(batchId_inEdge(:,1), :);
        ids2 = I(batchId_inEdge(:,2), :);

        data1 = res_pairwise(1).dzdx(:,:, 1 : numFeatures, :);
        data2 = res_pairwise(1).dzdx(:,:, numFeatures + 1 : 2 * numFeatures, :);
        
        data1 = reshape(data1, [], sum( numEdges ));
        data2 = reshape(data2, [], sum( numEdges ));
        
        featureBackPropSignal = featureBackPropSignal ...
            + reshape( data1 * ids1, size(featureBackPropSignal) ) ;
        featureBackPropSignal = featureBackPropSignal ...
            + reshape( data2 * ids2, size(featureBackPropSignal) ) ;

%         % SLOW VERSION
%         edgeOffset = 0;
%         for iImage = 1 : numImages
%             for iEdge = 1 : numEdges( iImage )
%                 instance1 = labels{iImage}.clusteredEdges.bbIds(iEdge, 1);
%                 instance1 = labels{iImage}.candidateBatchIds( instance1 );
%                 
%                 featureBackPropSignal(:,:,:,instance1) = featureBackPropSignal(:,:,:,instance1) ...
%                     + res_pairwise(1).dzdx(:,:, 1 : numFeatures, edgeOffset + iEdge);
%                 
%                 instance2 = labels{iImage}.clusteredEdges.bbIds(iEdge, 2);
%                 instance2 = labels{iImage}.candidateBatchIds( instance2 );
%                 
%                 featureBackPropSignal(:,:,:,instance2) = featureBackPropSignal(:,:,:,instance2) ...
%                     + res_pairwise(1).dzdx(:,:, numFeatures + 1 : 2 * numFeatures, edgeOffset + iEdge);
%             end
%             edgeOffset = edgeOffset + numEdges( iImage ) ;
%         end

        % backward pass of the pairwise signal through the feature extracting network
        [res_features, gradients] = vl_simplenn_pairwiseModel_backwardPass( net.featureExtractor, x, net.layers, res_features, gradients, featureBackPropSignal, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync, ...
            'disableDropout', disableFeatureExtractorDropout );
    end
end


end

function res = initRes(n)
res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
