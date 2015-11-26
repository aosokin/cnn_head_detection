function [minMarginals, bestLabeling, minMarginals_args] = computeMinMarginalsPairwiseBinary( unaryTerms, pairwiseTerms, varargin )
%computeMinMarginalsPairwiseBinary computes the min-marginals for the energy of unary and pairwise potentials and binary variables
%   if number of nodes is <= than 20 the computation is exact (see minimizeEnergyPairwiseBinary.m) otherwise approximations are used

if ~exist('varargin', 'var')
    varargin = {};
end

%% parameters
opts = struct;
opts.bigValue = 1e+4;
% parse input
opts = vl_argparse(opts, varargin);


%% check input
numLabels = 2;
if ~isnumeric(unaryTerms) || ~ismatrix(unaryTerms) || size(unaryTerms, 2) ~= 2
    error('Incorrect format for unaryTerms, has to be numNodes x 2')
end
numNodes = size(unaryTerms, 1);

if ~isnumeric(pairwiseTerms) || ~ismatrix(pairwiseTerms) || size(pairwiseTerms, 2) ~= 6
    error('Incorrect format for pairwiseTerms, has to be numEdges x 6')
end
numEdges = size(pairwiseTerms, 1);

%% compute the min-marginals
% run energy minimization without a loss
[bestLabeling, energy] = minimizeEnergyPairwiseBinary( unaryTerms, pairwiseTerms );

% compute min marginals
minMarginals = nan(numNodes, numLabels);
minMarginals( (1 : numNodes)' + numNodes * (bestLabeling - 1) ) = energy;

minMarginals_args = nan( numNodes, 2, numNodes );

for iNode = 1 : numNodes
    for iLabel = 1 : numLabels
        if iLabel ~= bestLabeling(iNode)
            curUnary = unaryTerms;
            curUnary(iNode, setdiff( 1 : numLabels, iLabel ) ) = opts.bigValue;
            [curLabels, curEnergy] = minimizeEnergyPairwiseBinary( curUnary, pairwiseTerms );
            
            if curLabels(iNode) ~= iLabel
                error('Min-marginal computation did not work!');
            end
            
            minMarginals(iNode, iLabel) = curEnergy;
            minMarginals_args( iNode, iLabel, : ) = reshape( curLabels, [1 1 numNodes] );
        else
            minMarginals_args( iNode, iLabel, : ) = reshape( bestLabeling, [1 1 numNodes] );
        end
    end
end

minMarginals_args = minMarginals_args - 1; % to be compatible with computeMinMarginalsBinaryMex.m

end

