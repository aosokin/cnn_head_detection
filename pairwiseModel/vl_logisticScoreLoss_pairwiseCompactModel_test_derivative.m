function vl_logisticScoreLoss_pairwiseCompactModel_test_derivative
%vl_logisticScoreLoss_pairwiseCompactModel_test_derivative tests vl_logisticScoreLoss_pairwiseCompactModel

% the .mat file is the same as for vl_svmStructLoss_pairwiseCompactModel_test_derivative.m
load vl_svmStructLoss_pairwiseCompactModel_test_derivative.mat unaryPotentials pairwisePotentials labels dzdy

[lossValue, unaryDerivative, pairwiseDerivative] = vl_logisticScoreLoss_pairwiseCompactModel( unaryPotentials, pairwisePotentials, labels, dzdy );

testEps = 1e-3;
numTrials = 100;
rng(1);

% test unary derivatives
empiricalUnaryDerivative = zeros(size(unaryPotentials), 'like', unaryPotentials);
fprintf('Number of unary derivatives: %d, testing %d\n', numel(empiricalUnaryDerivative), min(numTrials, numel(empiricalUnaryDerivative)));

randOrder = randperm( numel(empiricalUnaryDerivative), min(numTrials, numel(empiricalUnaryDerivative)) );
for iValueId = 1 : numel(randOrder)
    iValue = randOrder(iValueId);
    
    unaryNew = unaryPotentials;
    unaryNew(iValue) = unaryNew(iValue) + testEps;
    
    lossValue_test = vl_logisticScoreLoss_pairwiseCompactModel( unaryNew, pairwisePotentials, labels, []);
    
    empiricalUnaryDerivative(iValue) = sum(lossValue_test - lossValue) / testEps;
end

empDer = empiricalUnaryDerivative(randOrder);
trueDer = unaryDerivative(randOrder);
unaryDerivativeError = gather( norm(empDer(:) - trueDer(:)) / norm(trueDer(:)) );
fprintf('Relative error of unary derivative: %f\n', unaryDerivativeError );

% test pairwise derivatives
empiricalPairwiseDerivative = zeros(size(pairwisePotentials), 'like', pairwisePotentials);
fprintf('Number of pairwise derivatives: %d, testing %d\n', numel(empiricalPairwiseDerivative), min(numTrials, numel(empiricalPairwiseDerivative)));

randOrder = randperm( numel(empiricalPairwiseDerivative), min(numTrials, numel(empiricalPairwiseDerivative)) );
for iValueId = 1 : numel(randOrder)
    iValue = randOrder(iValueId);
    
    pairwiseNew = pairwisePotentials;
    pairwiseNew(iValue) = pairwiseNew(iValue) + testEps;
    
    lossValue_test = vl_logisticScoreLoss_pairwiseCompactModel( unaryPotentials, pairwiseNew, labels, []);
    
    empiricalPairwiseDerivative(iValue) = sum(lossValue_test - lossValue) / testEps;
end

empDer = empiricalPairwiseDerivative(randOrder);
trueDer = pairwiseDerivative(randOrder);
pairwiseDerivativeError = gather( norm(empDer(:) - trueDer(:)) / norm(trueDer(:)) );
fprintf('Relative error of pairwise derivative: %f\n', pairwiseDerivativeError );

end

