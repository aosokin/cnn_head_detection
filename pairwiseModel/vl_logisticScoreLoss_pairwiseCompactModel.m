function [lossValue_images, unaryDerivative, pairwiseDerivative, predictions] = vl_logisticScoreLoss_pairwiseCompactModel( unaryPotentials, pairwisePotentials, labels, dzdy, computeMaxMarginals )
%vl_logisticScoreLoss_pairwiseCompactModel implements the logistic loss on top of the structured scores
%
% The joint score:
%   S(y, theta) = \sum_i y_i * theta^U_i + \sum_ij y_i * y_j * \theta^P_{i,j,k_ij}
% where y_i \in \{0,1\} are the variables and theta^U, \theta_P - potentials
% i indexes the nodes, ij - the edges; k_ij - the cluster index of edge ij
%
% The individual scores are based on the max-marginals of the joint score:
%   s_i = \max_{ y: y_i = 1 } S(y, theta) - \max_{ y: y_i = 0 } S(y, theta)
%
% The final loss is computed as the sum of logistics on top of the individual scores:
%   loss = \sum_{i is positive} v( s_i ) + \sum_{i is negative} v( -s_i )
%
% If the number of nodes is <= 20 than the max-marginals are computed exacly using the exhaustive search, otherwise the approximations are used, see computeMinMarginalsPairwiseBinary.m
%
% Usage:
%   [lossValue_images, unaryDerivative, pairwiseDerivative, predictions] = vl_logisticScoreLoss_pairwiseCompactModel( unaryPotentials, pairwisePotentials, labels, dzdy, computeMaxMarginals )
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
    error('Unary potentials are of incorrect size');
end
numClusters = size(pairwisePotentials, 3);
if size(pairwisePotentials, 1) ~= 1 ||  size(pairwisePotentials, 2) ~= 1 || size(pairwisePotentials, 3) ~= numClusters * 1 || size(pairwisePotentials, 4) ~= sum(numEdges)
    error('Pairwise potentials are of incorrect size');
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
    
    % prepare terms for energy minimizationsc
    unaryTerms = [ -double(unaries(:)), zeros(numel(unaries), 1)];
    pairwiseTerms = [ edgeEnds, -double(pairwise(:)), zeros(numel(pairwise), 3) ];
    
    % labels{iImage}.classGroundTruth - 0 for bkg, 1 for obj
    % labels{iImage}.instanceGroundTruth - 0 for bkg, i for detection object #i, exactly one detection for object i should be present
    % labelsBinary - 1 - head, 2 - bkg
    
    % compute all the max-marginals
    if numNodes( iImage ) <= 20
        [minMarginals, minMarginals_args] = computeMinMarginalsBinaryMex( unaryTerms, pairwiseTerms );
    else
        [minMarginals, ~, minMarginals_args] = computeMinMarginalsPairwiseBinary( unaryTerms, pairwiseTerms );
    end
    maxMarginals = -minMarginals;
    
    scores = maxMarginals(:, 1) - maxMarginals(:, 2);
    [~, bestLabeling] = max( maxMarginals, [], 2 );
    
    % compute the logistic loss
    labelsBinary_GT = 2 * labels{iImage}.classGroundTruth - 1; % convert GT from (0 for bkg, 1 for obj) to (1 - head, -1 - bkg)
    expGtScores = exp( - scores .* labelsBinary_GT );
    
    numBkg = sum( labels{iImage}.classGroundTruth == 0 );
    numObj = sum( labels{iImage}.classGroundTruth == 1 );
    weightObj = 1 / numObj;
    weightBkg = 1 / numBkg;
    weights = weightBkg * ones( numNodes( iImage ), 1 );
    weights( labels{iImage}.classGroundTruth(:) == 1 ) = weightObj;
    weights = weights(:) / sum(weights(:)) * numel(weights);
    
    lossValue_images(iImage) = sum( log( 1 + expGtScores(:) ) .* weights(:) );
    
    if ~computeMaxMarginals
        predictions{iImage} = bestLabeling;
    else
        predictions{iImage}.maxMarginals = maxMarginals;
        predictions{iImage}.bestLabeling = bestLabeling;
    end
    
    if ~isempty(dzdy)
        % compute derivatives
        curUnaryDerivative = zeros(1, 1, 1, numNodes( iImage ), 'like', unaryDerivative);
        curPairwiseDerivative = zeros(1, 1, numClusters, numEdges( iImage ), 'like', pairwiseDerivative);
        
        % derivative of the loss w.r.t. the scores
        scoreDerivatives = (expGtScores(:) ./ (1 + expGtScores(:))) .* (-labelsBinary_GT(:)) .* weights(:);
        
        % derivative of the loss w.r.t. the max-marginals
        mmDerivates = [ ones( numNodes( iImage ), 1 ), -ones( numNodes( iImage ), 1 ) ];
        mmDerivates = bsxfun(@times, mmDerivates, scoreDerivatives);
        
        % derivative of the loss w.r.t. the potentials
        for iNode = 1 : numNodes( iImage )
            for iLabel = 1 : 2
                curArg = permute( minMarginals_args(iNode, iLabel, :), [3, 1, 2]);
                
                % unaries
                curMask = curArg(:) == 0; % the node belongs to the head class: 0 in this notaion
                curUnaryDerivative( curMask ) = curUnaryDerivative( curMask ) + 1 * mmDerivates(iNode, iLabel);
                
                % pairwise
                curMask = curArg( edgeEnds(:,1) ) == 0 & curArg( edgeEnds(:,2) ) == 0; % both incident nodes belong to the head class: 0 in this notaion
                curIds = edgeClusters(curMask) + numClusters * (find(curMask) - 1);
                curPairwiseDerivative( curIds ) = curPairwiseDerivative( curIds ) + 1 * mmDerivates(iNode, iLabel);
                
            end
        end
        
        unaryDerivative(:,:,:,nodeIds) = curUnaryDerivative;
        pairwiseDerivative(:,:,:,edgeIds) = curPairwiseDerivative;
    end
end

if ~isempty(dzdy)
    unaryDerivative = unaryDerivative * dzdy;
    pairwiseDerivative = pairwiseDerivative * dzdy;
end

end
