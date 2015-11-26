function [ unaryTermsNew, pairwiseTermsNew, energyConstant ] = projectEnergyBinaryPairwise( unaryTerms, pairwiseTerms, partialLabels )
%projectEnergyBinaryPairwise assigns spesified values to the subset of variables and constructs the energy from the unlabelled ones

maskUnlabeled = partialLabels == 0;
numUnlabeled = sum(maskUnlabeled);
numNodes = size( unaryTerms, 1 );
numEdges = size( pairwiseTerms, 1);
numLabels = 2;

maskEdgesNew = maskUnlabeled( pairwiseTerms(:, 1) ) & maskUnlabeled( pairwiseTerms(:, 2) );

pairwiseTermsNew = pairwiseTerms(maskEdgesNew, :);
newIds = nan(numUnlabeled, 1);
newIds(maskUnlabeled) = 1 : numUnlabeled;
pairwiseTermsNew(:, 1) = newIds( pairwiseTermsNew(:, 1) );
pairwiseTermsNew(:, 2) = newIds( pairwiseTermsNew(:, 2) );

energyConstant = 0;
unaryTermsNew = unaryTerms(maskUnlabeled, :); 
labeledUnary = unaryTerms(~maskUnlabeled, :);
goodLabels = partialLabels(~maskUnlabeled);
energyConstant = energyConstant + ...
    sum( labeledUnary( (1 : numNodes - numUnlabeled)' + (goodLabels - 1) * (numNodes - numUnlabeled) ) );

labelMap = [ 3, 4; 5, 6];

for iEdge = 1 : numEdges
    node1 = pairwiseTerms(iEdge, 1);
    node2 = pairwiseTerms(iEdge, 2);
    label1 = partialLabels(node1);
    label2 = partialLabels(node2);
    if ~maskUnlabeled(node1) && ~maskUnlabeled(node2)
        energyConstant = energyConstant + pairwiseTerms( iEdge, labelMap(label1, label2) );
    elseif maskUnlabeled(node1) && ~maskUnlabeled(node2)
        unaryTermsNew( newIds(node1), : ) = unaryTermsNew( newIds(node1), : ) + ...
            reshape( pairwiseTerms(iEdge, labelMap(:, label2)), 1, numLabels);
    elseif ~maskUnlabeled(node1) && maskUnlabeled(node2)
        unaryTermsNew( newIds(node2), : ) = unaryTermsNew( newIds(node2), : ) + ...
            reshape( pairwiseTerms(iEdge, labelMap(label1, :)), 1, numLabels);
    end
end


end

