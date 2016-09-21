function [net, info] = cnn_train_pairwiseModel(net, imdb, getBatch, varargin)
%cnn_train_pairwiseModel trains the CNN-based pairwise model using MatConvNet
%cnn_train_pairwiseModel is the modified version of MatConvNet's cnn_train.m

opts.train = [] ;
opts.val = [] ;
opts.lossNormalization = 1;
opts.disableDropoutFeatureExtractor = false;
opts.numEpochs = 1 ;
opts.restartEpoch = nan;
opts.backPropagateType = 'all'; % 'all', 'unaryAndPairwise', 'onlyUnary', 'onlyPairwise'
opts.batchSize = 256 ;
opts.learningRate = 0.001 ;
opts.continue = true ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.numValidationPerEpoch = 1; % how often to call validation: useful is training set is super large
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for l=1:numel(net.layers)
    if isfield(net.layers{l}, 'weights')
        J = numel(net.layers{l}.weights) ;
        for j=1:J
            net.layers{l}.momentum{j} = zeros(size(net.layers{l}.weights{j}), 'single') ;
        end
        if ~isfield(net.layers{l}, 'learningRate')
            net.layers{l}.learningRate = ones(1, J, 'single') ;
        end
        if ~isfield(net.layers{l}, 'weightDecay')
            net.layers{l}.weightDecay = ones(1, J, 'single') ;
        end
    end
end

net = vl_simplenn_move(net, 'gpu') ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

rng(0) ;

one = gpuArray(single(1)) ;

info.train.objective = [] ;
info.train.error = [] ;
info.train.speed = [] ;
info.train.datasetSize = numel(opts.train);
info.train.xPos = [];
info.val.objective = [] ;
info.val.error = [] ;
info.val.speed = [] ;
info.val.datasetSize = numel(opts.val);
info.val.xPos = [];

validationBatches = ceil( numel(opts.train) / opts.batchSize );
if ~isnan( opts.numValidationPerEpoch )
    validationBatches = ceil( numel(opts.train) / opts.batchSize * (1 : opts.numValidationPerEpoch) / opts.numValidationPerEpoch );
    
    fprintf('Validation will be performed %d times each training epoch\n', length(validationBatches) );
end

modelPath = fullfile(opts.expDir, 'net-epoch-%d-%d.mat') ;
if ~isnan(opts.restartEpoch)
    if ~exist(sprintf(modelPath, opts.restartEpoch),'file')
        opts.restartEpoch = nan;
        warning('File for epoch %d was not found!, restarting standard procedure');
    end
end

lr = 0 ;
res = [] ;
gradients = [];

flagStartedTraining = false;
for epoch=1:opts.numEpochs
    prevLr = lr ;
    lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    
    % fast-forward to where we stopped
    modelPath = @(ep, number) fullfile(opts.expDir, sprintf('net-epoch-%d-%d.mat', ep, number));
    modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
    if opts.continue && ~flagStartedTraining
        % check if can fast forward to the target epoch
        if ~isnan(opts.restartEpoch) && epoch <= opts.restartEpoch
            if exist( modelPath(opts.restartEpoch, opts.numValidationPerEpoch),'file')
                continue;
            end
        elseif exist( modelPath( epoch, opts.numValidationPerEpoch ),'file' )
            continue;
        end
        
        if epoch > 1
            fprintf('resuming by loading epoch %d\n', epoch-1) ;
            load(modelPath(epoch-1, opts.numValidationPerEpoch), 'net', 'info') ;
        end
    end
    flagStartedTraining = true;
    
    train = opts.train(randperm(numel(opts.train))) ;
    
    info.train.objective(end+1) = 0 ;
    info.train.error(end+1) = 0 ;
    info.train.speed(end+1) = 0 ;
    
    % reset momentum if needed
    if prevLr ~= lr
        fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
        for l=1:numel(net.layers)
            if isfield(net.layers{l}, 'weights')
                for j = 1 : length( net.layers{l}.weights )
                    net.layers{l}.momentum{j} = 0 * net.layers{l}.momentum{j};
                end
            end
        end
    end
    
    numTrain = 0;
    numSavedModels = 0;
    for t=1:opts.batchSize:numel(train)
        % get next image batch and labels
        batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
        batch_time = tic ;
        fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix((t-1)/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
        [im, labels] = getBatch(imdb, batch) ;
        batchSize = numel(batch);
        numTrain = numTrain + batchSize;
        
        im = gpuArray(im) ;
        
        % backprop
        [lossValue, gradients, predictions] = vl_structuredNetwork_pairwiseModel(net, im, gradients, labels, one, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', true, ...
            'backPropagateType', opts.backPropagateType, ...
            'disableDropoutFeatureExtractor', opts.disableDropoutFeatureExtractor ) ;
        
        % gradient step
        for l = 1 : numel(net.layers)
            if isfield(net.layers{l}, 'weights')
                for j = 1 : length( net.layers{l}.weights )
                    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
                    thisLR = lr * net.layers{l}.learningRate(j) ;
                    
                    net.layers{l}.momentum{j} = ...
                        opts.momentum * net.layers{l}.momentum{j} ...
                        - thisDecay * net.layers{l}.weights{j} ...
                        - (1 / batchSize) * gradients{l}.dzdw{j} ;
                    net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j} ;
                end
            end
        end
        
        % print information
        batch_time = toc(batch_time) ;
        speed = batchSize/batch_time ;
        info.train = updateError( info.train, lossValue, predictions, labels, batch_time) ;
        
        fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
        fprintf(' err %.6f loss %.6f', ...
            info.train.error(end)/numTrain, info.train.objective(end)/numTrain) ;
        fprintf('\n') ;
        
        
        % run validation if the batch number is in the validationBatches
        if ismember( (t - 1) / opts.batchSize + 1, validationBatches)
            % evaluation on validation set
            info.val.objective(end+1) = 0 ;
            info.val.error(end+1) = 0 ;
            info.val.speed(end+1) = 0 ;
            info.val.xPos(end+1) = epoch-1 + min(t+opts.batchSize-1, numel(train)) / numel(train);
            
            info = evaluateValidationSet( opts, imdb, net, epoch, info, getBatch );
            
            % save the intermediate model
            save(modelPath(epoch, numSavedModels + 1), 'net', 'info') ;
            numSavedModels = numSavedModels + 1;
            
            makePlots(info, modelFigPath);
        end
        
        
    end % next batch
    
    % save
    info.train.objective(end) = info.train.objective(end) / numTrain ;
    info.train.error(end) = info.train.error(end) / numTrain  ;
    info.train.speed(end) = numTrain / info.train.speed(end) ;
    info.train.xPos(end+1) = epoch;
    if numSavedModels == 0
        numSavedModels = numSavedModels + 1;
    end
    save(modelPath(epoch, numSavedModels), 'net', 'info') ;
    
    makePlots(info,  modelFigPath);
end

% save final model
for i = 1 : length(net.layers)
    if isequal( net.layers{i}.type, 'conv' )
        for j = 1 : length( net.layers{i}.weights )
            net.layers{i}.weights{j} = single( gather( net.layers{i}.weights{j} ) );
        end
        net.layers{i} = rmfield( net.layers{i}, 'momentum' );
    end
end
save( fullfile( opts.expDir, 'pairwise.mat' ), '-struct', 'net', '-v7.3' );

end

function makePlots(info, modelFigPath)
figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(info.train.xPos, info.train.objective(1:numel(info.train.xPos)), 'k') ; hold on ;
semilogy( info.val.xPos, info.val.objective(1:numel(info.val.xPos)), 'b') ;
xlabel('training epoch') ; ylabel('energy') ;
grid on ;
h=legend('train', 'val') ;
set(h,'color','none');
title('objective') ;

subplot(1,2,2) ;
plot(info.train.xPos, info.train.error(1:numel(info.train.xPos)), 'k') ; hold on ;
plot(info.val.xPos, info.val.error(1:numel(info.val.xPos)), 'b') ;
h=legend('train','val') ;
grid on ;
xlabel('training epoch') ; ylabel('error') ;
set(h,'color','none') ;
title('error') ;
drawnow ;
print(1, modelFigPath, '-dpdf') ;
end

function lossValue = computeDetectionLoss(predictions, labels)
numImages = length(predictions);
if length(labels) ~= numImages
    error('computeDetectionLoss:labelGtNumInstanceMismatch', 'GT and predictions are provided for different number of instances');
end

lossValue = zeros(numImages, 1);

for iImage = 1 : numImages
    lossValue(iImage) = 0;
    
    curBbMatch = labels{iImage}.instanceGroundTruth;
    if numel(curBbMatch) ~= numel(predictions{iImage})
        error('computeDetectionLoss:labelGtLengthMismatch', 'Something predictions and the ground truth are not compatible');
    end
    
    lossValue(iImage) = lossValue(iImage) + sum(curBbMatch(:) == 0 & predictions{iImage}(:) == 1); % penalty for each background BB detected
    
    existingObjects = unique(curBbMatch);
    for iObjectId = 1 :length( existingObjects )
        iObject = existingObjects( iObjectId );
        if iObject ~= 0
            lossValue(iImage) = lossValue(iImage) + abs( sum(curBbMatch(:) == iObject & predictions{iImage}(:) == 1) - 1 ); % each object in the GT has to be detected exactly once
        end
    end
end

end

function info = updateError(info, objectiveValue, predictions, labels, speed)
info.objective(end) = info.objective(end) + sum(double(gather(objectiveValue)));
info.speed(end) = info.speed(end) + speed ;

errorValue = computeDetectionLoss(predictions, labels);
info.error(end) = info.error(end) + sum(errorValue(:));
end

function info = evaluateValidationSet( opts, imdb, net, epoch, info, getBatch )
val = opts.val;
numVal = 0;
for t=1:opts.batchSize:numel(val)
    batch_time = tic ;
    batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...', epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    [im, labels] = getBatch(imdb, batch) ;
    batchSize = numel(batch);
    numVal = numVal + batchSize;
    
    im = gpuArray(im) ;
    
    [lossValue, ~, predictions] = vl_structuredNetwork_pairwiseModel(net, im, [], labels, [], ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', true, ...
        'disableDropoutFeatureExtractor', opts.disableDropoutFeatureExtractor ) ;
    
    % print information
    batch_time = toc(batch_time) ;
    speed = batchSize/batch_time ;
    
    info.val = updateError(info.val, lossValue, predictions, labels, batch_time) ;
    
    fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    fprintf(' err %.3f loss %.3f ', ...
        info.val.error(end)/numVal, mean(lossValue) ) ;
    fprintf('\n') ;
end

info.val.objective(end) = info.val.objective(end) / numVal ;
info.val.error(end) = info.val.error(end) / numVal ;
info.val.speed(end) = numVal / info.val.speed(end) ;


end


