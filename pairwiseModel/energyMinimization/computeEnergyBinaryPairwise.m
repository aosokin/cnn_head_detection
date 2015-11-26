function energy = computeEnergyBinaryPairwise( unaryTerms, pairwiseTerms, labels )
%computeEnergyBinaryPairwise computes the value of the energy with unary and pairwise potentials

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

if ~isnumeric(labels) || ~isvector(labels) || length(labels) ~= numNodes
    error('Incorrect format for labels, has to be numNodes x 1')
end
labels = labels(:);
if any(labels > numLabels) || any(labels < 1)
    error('Incorrect values for labels, has to be an integer from 1 to numLabels')
end

%% computation
energy = sum( unaryTerms((1 : numNodes)' + numNodes * (labels - 1) ) );

label1 = labels(pairwiseTerms(:, 1));
label2 = labels(pairwiseTerms(:, 2));
labelMap = [ 3, 4; 5, 6];
jointLabelMap = labelMap( label1 + 2 * (label2 - 1) );

energy = energy + sum( pairwiseTerms( (1 : numEdges)' + numEdges * (jointLabelMap - 1) ) );

end

