function [lossValue_images, unaryDerivative, pairwiseDerivative, predictions] = vl_svmStructLoss_pairwiseCompactModel( unaryPotentials, pairwisePotentials, labels, dzdy, computeMaxMarginals, lossNormalization )
%vl_svmStructLoss_pairwiseCompactModel implements the structured SVM loss
%
% The joint score:
%   S(y, theta) = \sum_i y_i * theta^U_i + \sum_ij y_i * y_j * \theta^P_{i,j,k_ij}
% where y_i \in \{0,1\} are the variables and theta^U, \theta_P - potentials
% i indexes the nodes, ij - the edges; k_ij - the cluster index of edge ij
%
% The loss max-margin loss
%   loss = \max_\hat{y}( S(\hat{y}, theta) + \Delta(\hat{y}, y) ) - S(y, theta)
% where y is the ground-truth labeling, and \Delta is the Hamming loss with  the balancing of the class penalties
%
% If the number of nodes is <= 20 than the maximization is done exactly, otherwise the approximations are used, see minimizeEnergyPairwiseBinary.m
%
% Usage:
%   [lossValue_images, unaryDerivative, pairwiseDerivative, predictions] = vl_svmStructLoss_pairwiseCompactModel( unaryPotentials, pairwisePotentials, labels, dzdy, computeMaxMarginals, lossNormalization )
%
% Input:
%   unaryPotentials - unary potentials parameterized by \theta^U, double[1 x 1 x 1 x numNodes], where numNodes is the number of all nodes in the batch
%   pairwisePotentials - pairwise potentials parameterizws by \theta^P, double[1 x 1 x numClusters x numNodes]
%   labels - cell array providing information about the batch, cell[numImages x 1], each cell has
%       candidateBatchIds - indices of the nodes of the current image, double[ numNodesInImage x 1 ]
%       classGroundTruth - class labels of the nodes of the current image, double[ numNodesInImage x 1 ] with labels 0 or 1
%       instanceGroundTruth - indices of the gound-truth objects of the current image, double[ numNodesImage x 1 ] of 0 (background), 1, ..., numInstances
%       clusteredEdges - information about clusters of the edges, structure with fields:
%           bbIds - indices of the nodes connected with an edge, double[numEdges x 2] of 1,...,numNodesInImage
%           clusterId - IDs of clusters for the edges, double[numEdges x 1]
%   dzdy - if empty, only forward pass is computed, otherwise the gradient is multiplied by dzdy
%   computeMaxMarginals - flag showing whether to compute output max-marginals (default: false)
%       CAUTION: the function supports either the output of the max-marginals or the output of the gradient (because different optimization problems have to be solved)
%   lossNormalization - normalization constant for the loss (default: 1)
%
% Output: 
%   lossValue_images - values of the loss for all images in a batch, double[numImages x 1]
%   unaryDerivative - derivatives w.r.t. the \theta^U, same size as unaryPotentials
%   pairwiseDerivative - derivatives w.r.t. the \theta^P, same size as pairwisePotentials
%   predictions - if computeMaxMarginals == false than labels for the nodes, double[numNodesImage x 1]
%           if computeMaxMarginals == true than cell[numImages x 1], each cell has
%              bestLabeling - labels for the nodes, double[numNodesImage x 1]
%              maxMarginals - max-marginals

if ~exist('computeMaxMarginals', 'var') || isempty(computeMaxMarginals)
    computeMaxMarginals = false;
end

if ~exist('lossNormalization', 'var') || isempty(lossNormalization)
    lossNormalization = 1; % the loss is rescaled to [0, lossNormalization] segment
end

%% preparation
numImages = length(labels);
numNodes = zeros(numImages, 1);
numEdges = zeros(numImages, 1);
numLabels = 2;
for iImage = 1 : numImages
    numNodes( iImage ) = length( labels{iImage}.candidateBatchIds );
    numEdges( iImage ) = size(labels{ iImage }.clusteredEdges.bbIds, 1);
end

if size(unaryPotentials, 1) ~= 1 ||  size(unaryPotentials, 2) ~= 1 || size(unaryPotentials, 3) ~= 1 || size(unaryPotentials, 4) ~= sum(numNodes)
    error('vl_svmStructLoss_pairwiseCompactModel:badUnaries', 'Unary potentials are of incorrect size');
end
numClusters = size(pairwisePotentials, 3);
if size(pairwisePotentials, 1) ~= 1 ||  size(pairwisePotentials, 2) ~= 1 || size(pairwisePotentials, 3) ~= numClusters * 1 || size(pairwisePotentials, 4) ~= sum(numEdges)
    error('vl_svmStructLoss_pairwiseCompactModel:badPairwise', 'Pairwise potentials are of incorrect size');
end

if ~isempty(dzdy)
    % compute derivatives
    unaryDerivative = zeros( size(unaryPotentials), 'like', unaryPotentials );
    pairwiseDerivative = zeros( size(pairwisePotentials), 'like', pairwisePotentials );
else
    unaryDerivative = [];
    pairwiseDerivative = [];
end

%% start computations
nodeStartIndex = 0;
edgeStartIndex = 0;
lossValue_images = nan(numImages, 1);
predictions = cell(numImages, 1);
for iImage = 1 : numImages
    %% extract potentials
    nodeIds = nodeStartIndex + (1 : numNodes( iImage ));
    edgeIds = edgeStartIndex + (1 : numEdges( iImage ));
    nodeStartIndex = nodeStartIndex + numNodes( iImage );
    edgeStartIndex = edgeStartIndex + numEdges( iImage );
    
    % extract unaries
    unaries = reshape( unaryPotentials(:,:,:,nodeIds), 1, numNodes( iImage ) );
    
    % extract pairwise potentials by mapping only one cluster to the edge
    edgeEnds = labels{ iImage }.clusteredEdges.bbIds;
    edgeClusters = labels{ iImage }.clusteredEdges.clusterId;
    
    pairwise = pairwisePotentials(:,:,:,edgeIds);
    
    potentialsId = edgeClusters(:) + numClusters * (0 : numEdges( iImage ) - 1)';
    pairwise = reshape( pairwise( potentialsId ), 1, numEdges( iImage ) );
    
    % prepare terms for energy minimization
    unaryTerms = [ -double(unaries(:)), zeros(numel(unaries), 1)];
    pairwiseTerms = [ edgeEnds, -double(pairwise(:)), zeros(numel(pairwise), 3) ];
    
    % labels{iImage}.classGroundTruth - 0 for bkg, 1 for obj
    % labels{iImage}.instanceGroundTruth - 0 for bkg, i for detection object #i, exactly one detection for object i should be present
    % labelsBinary - 1 - head, 2 - bkg
    
    % produce the output labelling
    if ~computeMaxMarginals
        % output the result of the loss-augmented inference and proceed to the gradient computation
        
        % get the weights for the objects
        numBkg = sum( labels{iImage}.classGroundTruth == 0 );
        numObj = sum( labels{iImage}.classGroundTruth == 1 );
        weightObj = 1 / numObj;
        weightBkg = 1 / numBkg;
        weights = weightBkg * ones( numNodes( iImage ), 1 );
        weights( labels{iImage}.classGroundTruth(:) == 1 ) = weightObj;
        weights = weights(:) / sum(weights(:));  % the weights sum to one
    
        % update the unaries with the loss
        labelsBinary_GT = 1 + ( 1 - labels{iImage}.classGroundTruth ); % convert GT from (0 for bkg, 1 for obj) to (1 - head, 2 - bkg)
        I = ones(numLabels) - eye(numLabels);
        lossUpdate = bsxfun( @times, I(labelsBinary_GT, :), weights ) * lossNormalization; % the loss updates are weighted by the weights
        unaryTerms_lossAugmented = unaryTerms - lossUpdate;
        
        [labelsBinary_worst, energyWorst, isOptimal] = minimizeEnergyPairwiseBinary( unaryTerms_lossAugmented, pairwiseTerms, ...
            'labelingToCheck', 1+(1-labels{iImage}.classGroundTruth) );
        energyWorst = -energyWorst;
        
        predictions{iImage} = labelsBinary_worst;
        
        % compute energy at the ground truth
        energyGt = computeEnergyBinaryPairwise( unaryTerms, pairwiseTerms, 1+(1-labels{iImage}.classGroundTruth) );
        energyGt = -energyGt;
    
        lossValue_images(iImage) = energyWorst - energyGt;
    
    else
        % output the min marginals: not possible to do the gradient computation
        if ~isempty(dzdy)
            error('vl_svmStructLoss_pairwiseCompactModel:ambiguosMode', 'This function either computes the min-marginals or the gradient');
        end
        [ minMarginals, bestLabeling ] = computeMinMarginalsPairwiseBinary( unaryTerms, pairwiseTerms );
       
        predictions{iImage}.maxMarginals = -minMarginals;
        predictions{iImage}.bestLabeling = bestLabeling;
    end
    
    if ~isempty(dzdy)
        % compute derivatives
        curUnaryDerivative = zeros(1, 1, 1, numNodes( iImage ), 'like', unaryDerivative);
        curPairwiseDerivative = zeros(1, 1, numClusters, numEdges( iImage ), 'like', pairwiseDerivative);
        
        % unaries
        curMask = labelsBinary_worst(:) == 1;
        curUnaryDerivative( curMask ) = curUnaryDerivative( curMask ) + 1;
        curMask = labelsBinary_GT(:) == 1;
        curUnaryDerivative( curMask ) = curUnaryDerivative( curMask ) - 1;
        
        % pairwise;
        curMask = labelsBinary_worst( edgeEnds(:,1) ) == 1 & labelsBinary_worst( edgeEnds(:,2) ) == 1;
        curIds = edgeClusters(curMask) + numClusters * (find(curMask) - 1); 
        curPairwiseDerivative( curIds ) = curPairwiseDerivative( curIds ) + 1;
        
        curMask = labelsBinary_GT( edgeEnds(:,1) ) == 1 & labelsBinary_GT( edgeEnds(:,2) ) == 1;
        curIds = edgeClusters(curMask) + numClusters * (find(curMask) - 1); 
        curPairwiseDerivative( curIds ) = curPairwiseDerivative( curIds ) - 1;
        
        unaryDerivative(:,:,:,nodeIds) = curUnaryDerivative;
        pairwiseDerivative(:,:,:,edgeIds) = curPairwiseDerivative;
    end
end

if ~isempty(dzdy)
    unaryDerivative = unaryDerivative * dzdy;
    pairwiseDerivative = pairwiseDerivative * dzdy;
end

end
