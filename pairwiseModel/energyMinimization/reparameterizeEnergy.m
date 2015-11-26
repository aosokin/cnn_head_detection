function [ unary, pairwise, constant ] = reparameterizeEnergy( unaryTerms, pairwiseTerms )
%reparameterizeEnergy performs the reparametrization of the energy such that it is represented as 
% \sum_i x_i + \sum_{ij} x_i * x_j, x_i, x_j \in \{0, 1\}

numNodes = size( unaryTerms, 1 );
numEdges = size( pairwiseTerms, 1 );
numLabels = 2;
if size(unaryTerms, 2) ~= 2
    error( 'This function is implemented for binary variables only' );
end

pairwise = nan(numEdges, 3);
pairwise(:, 1) = pairwiseTerms(:, 1);
pairwise(:, 2) = pairwiseTerms(:, 2);
pairwise(:, 3) = pairwiseTerms(:, 3) + pairwiseTerms(:, 6) ...
    -pairwiseTerms(:, 4) - pairwiseTerms(:, 5);

unary = zeros(numNodes, 1);
% compensate for x_i
unary = updateVector(unary, pairwiseTerms(:,1), ...
    pairwiseTerms(:,5) - pairwiseTerms(:,3) );
% compensate for x_4
unary = updateVector(unary, pairwiseTerms(:,2), ...
    pairwiseTerms(:,4) - pairwiseTerms(:,3) );

constant = sum( pairwiseTerms(:,3) );

% add initial unaries
unary = unary + unaryTerms(:, 2) - unaryTerms(:, 1);
constant = constant + sum( unaryTerms(:, 1) );
end

function x = updateVector( x, index, update )
% x(index) = x(index) + update; 
% does not work well when there are matches in the index vector

updateSummed = accumarray(index, update, [length(x), 1], @sum);
x = x + updateSummed;
end
