function vl_structuredNetwork_pairwiseModel_test_derivative
%vl_structuredNetwork_pairwiseModel_test_derivative tests vl_structuredNetwork_pairwiseModel

% create the following file from the variables of cnn_train_pairwiseModel.m right before the call of vl_structuredNetwork_pairwiseModel.m
load( 'vl_structuredNetwork_pairwiseModel_test_derivative.mat', 'net', 'im', 'gradients', 'labels', 'one' );
testEps = 1e-3;

fprintf('Computing the gradient ... ');
tStart = tic;
[lossValue, gradients, predictions] = vl_structuredNetwork_pairwiseModel(net, im, gradients, labels, one, ...
    'conserveMemory', true, ...
    'sync', true, ...
    'disableDropout', true ) ;
fprintf( '%f\n', toc(tStart) );

maxGroupTests = 100;
rng(1);

% test derivatives
for iLayer = length( net.layers ) : -1 : 1
    if ~isequal( net.layers{iLayer}.type, 'conv' )
        continue;
    end
    fprintf('Layer %d: %s\n', iLayer, net.layers{iLayer}.name);
    
    empiricalDerivative = zeros( numel(gradients{ iLayer }.dzdw{2}), 1, 'like', gradients{ iLayer }.dzdw{2});
    fprintf('Number of bias derivatives: %d\n', numel(empiricalDerivative));
    
    randOrder = randperm(numel(empiricalDerivative));
    numTests = min( numel(empiricalDerivative), maxGroupTests);
    for iValueIndex = 1 : numTests
        if mod(iValueIndex, 1000) == 0
            fprintf('Derivative #%d\n', iValueIndex);
        end
        iValue = randOrder(iValueIndex);
        
        initValue = net.layers{iLayer}.weights{2}(iValue);
        net.layers{iLayer}.weights{2}(iValue) = initValue + testEps;
        
        [lossValue_test, ~, predictions_test] = vl_structuredNetwork_pairwiseModel(net, im, gradients, labels, [], ...
            'conserveMemory', true, ...
            'sync', true, ...
            'disableDropout', true) ;
        
        empiricalDerivative(iValue) = sum(lossValue_test - lossValue) / testEps;
        
        net.layers{iLayer}.weights{2}(iValue) = initValue;
    end
    
    testIndices = randOrder(1 : numTests);
    emphiricalGradient = empiricalDerivative(testIndices);
    computedGradient = gradients{ iLayer }.dzdw{2}(testIndices);
    derivativeError = gather( norm(emphiricalGradient(:) - computedGradient(:)) / norm(computedGradient(:)) );
    
    fprintf('Relative error of bias derivatives: %f\n', derivativeError );
    fprintf('Norm of tested derivatives: %f\n', norm(computedGradient(:)) );
    
    
    
    empiricalDerivative = zeros( numel(gradients{ iLayer }.dzdw{1}), 1, 'like', gradients{ iLayer }.dzdw{1});
    fprintf('Number of filter derivatives: %d\n', numel(empiricalDerivative));
    
    randOrder = randperm(numel(empiricalDerivative));
    numTests = min( numel(empiricalDerivative), maxGroupTests);
    for iValueIndex = 1 : numTests
        if mod(iValueIndex, 1000) == 0
            fprintf('Derivative #%d\n', iValueIndex);
        end
        
        iValue = randOrder(iValueIndex);
        
        initValue = net.layers{iLayer}.weights{1}(iValue);
        net.layers{iLayer}.weights{1}(iValue) = initValue + testEps;
        
        [lossValue_test, ~, predictions_test] = vl_structuredNetwork_pairwiseModel(net, im, gradients, labels, [], ...
            'conserveMemory', true, ...
            'sync', true, ...
            'disableDropout', true) ;
        
        empiricalDerivative(iValue) = sum(lossValue_test - lossValue) / testEps;
        
        net.layers{iLayer}.weights{1}(iValue) = initValue;
    end
    
    testIndices = randOrder(1 : numTests);
    emphiricalGradient = empiricalDerivative(testIndices);
    computedGradient = gradients{ iLayer }.dzdw{1}(testIndices);
    derivativeError = gather( norm(emphiricalGradient(:) - computedGradient(:)) / norm(computedGradient(:)) );
    
    fprintf('Relative error of filter derivatives: %f\n', derivativeError );
    fprintf('Norm of tested derivatives: %f\n', norm(computedGradient(:)) );
    
end


end

