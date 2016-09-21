function [net, info] = cnn_train_globalModel(net, imdb, getBatch, varargin)
%cnn_train_globalModel trains the CNN using MatConvNet
%cnn_train_globalModel is the modified version of MatConvNet's cnn_train.m

opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.restartEpoch = nan;

opts.batchSize = 256 ;
opts.useGpu = true ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.sync = true ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.backPropDepth = +inf;
opts.plotDiagnostics = false ;
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

if opts.useGpu
    net = vl_simplenn_move(net, 'gpu') ;
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

rng(0) ;

if opts.useGpu
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

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
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
        [im, labels] = getBatch(imdb, batch) ;
        batchSize = size(im, 4);
        numTrain = numTrain + batchSize;
        
        if opts.useGpu
            im = gpuArray(im) ;
        end
        
        % backprop
        net.layers{end}.class = labels ;
        res = vl_simplenn_globalModel(net, im, one, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync, ...
            'backPropDepth', opts.backPropDepth);
        
        % gradient step
        for l = max(1, numel(net.layers) - opts.backPropDepth + 1) : numel(net.layers)
            if isfield(net.layers{l}, 'weights')
                for j = 1 : length( net.layers{l}.weights )
                    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
                    thisLR = lr * net.layers{l}.learningRate(j) ;
                
                    net.layers{l}.momentum{j} = ...
                        opts.momentum * net.layers{l}.momentum{j} ...
                        - thisDecay * net.layers{l}.weights{j} ...
                        - (1 / batchSize) * res(l).dzdw{j} ;
                    net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j} ;
                end
            end
        end
        
        % print information
        batch_time = toc(batch_time) ;
        speed = batchSize/batch_time ;
        info.train = updateError(info.train, labels, res, batch_time) ;
        
        fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
        fprintf(' err %.3f loss %.3f ', info.train.error(end)/numTrain*100,  sum(double(gather(res(end).x))));
        fprintf('\n') ;
        
        % run validation if the batch number is in the validationBatches
        if ismember( (t - 1) / opts.batchSize + 1, validationBatches)
            % evaluation on validation set
            info.val.objective(end+1) = 0 ;
            info.val.error(end+1) = 0 ;
            info.val.speed(end+1) = 0 ;
            info.val.xPos(end+1) = epoch-1 + min(t+opts.batchSize-1, numel(train)) / numel(train);
            
            info = evaluateValidationSet( opts, imdb, net, epoch, info, getBatch, res  );

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
    if numSavedModels == 0
        numSavedModels = numSavedModels + 1;
    end
    save(modelPath(epoch, numSavedModels), 'net', 'info') ;
    
    makePlots(info, modelFigPath);
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
save( fullfile( opts.expDir, 'global.mat' ), '-struct', 'net', '-v7.3' );


end

function makePlots(info, modelFigPath)
figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(info.train.xPos, info.train.objective(1:numel(info.train.xPos)), 'k') ; hold on ;
semilogy(info.val.xPos, info.val.objective(1:numel(info.val.xPos)), 'b') ;
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

function info = updateError(info, labels, res, speed)
predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

info.objective(end) = info.objective(end) + sum(double(gather(res(end).x))) ;
info.speed(end) = info.speed(end) + speed ;

[~,predictions] = sort(predictions, 3, 'descend') ;
error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
info.error(end) = info.error(end) +....
    sum(sum(sum(error(:,:,1,:))))/n ;
end


function info = evaluateValidationSet( opts, imdb, net, epoch, info, getBatch, res  )
val = opts.val;
numVal = 0;
for t=1:opts.batchSize:numel(val)
    batch_time = tic ;
    batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...', epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    [im, labels] = getBatch(imdb, batch) ;
    batchSize = size(im, 4);
    numVal = numVal + batchSize;
    if opts.prefetch
        nextBatch = val(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(val))) ;
        getBatch(imdb, nextBatch) ;
    end
    if opts.useGpu
        im = gpuArray(im) ;
    end
    
    net.layers{end}.class = labels ;
    res = vl_simplenn_globalModel(net, im, [], res, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync );
    
    % print information
    batch_time = toc(batch_time) ;
    speed = batchSize/batch_time ;
    info.val = updateError(info.val, labels, res, batch_time) ;
    
    fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    n = numVal;
    fprintf(' err %.1f ', info.val.error(end)/n*100) ;
    fprintf('\n') ;
end

info.val.objective(end) = info.val.objective(end) / numVal ;
info.val.error(end) = info.val.error(end) / numVal ;
info.val.speed(end) = numVal / info.val.speed(end) ;
end
