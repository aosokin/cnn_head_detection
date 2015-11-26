function [res, gradients] = vl_simplenn_pairwiseModel_backwardPass(net, x, trainableLayers, res, gradients, dzdy, varargin)
%vl_simplenn_pairwiseModel_backwardPass performs the backward pass using the prodided CNN

opts = struct;
opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.doder = false;
opts.backPropDepth = +inf;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

gpuMode = isa(x, 'gpuArray') ;

res(n+1).dzdx = dzdy ;
for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
        case 'conv'
            if ~opts.accumulate
                [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                    vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                    res(i+1).dzdx, ...
                    'pad', l.pad, 'stride', l.stride) ;
            else
                dzdw = cell(1,2) ;
                [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                    vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                    res(i+1).dzdx, ...
                    'pad', l.pad, 'stride', l.stride) ;
                for j=1:2
                    res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
                end
                clear dzdw ;
            end
            
        case 'convPtr'
            id = l.index;
            [res(i).dzdx, filterGradients, biasGradients] = ...
                vl_nnconv(res(i).x, trainableLayers{id}.weights{1}, trainableLayers{id}.weights{2}, ...
                res(i+1).dzdx, ...
                'pad', trainableLayers{id}.pad, 'stride', trainableLayers{id}.stride) ;
            gradients{id}.dzdw{1} = gradients{id}.dzdw{1} + filterGradients;
            gradients{id}.dzdw{2} = gradients{id}.dzdw{2}  + biasGradients;
            
        case 'pool'
            res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
        case 'normalize'
            res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
        case 'softmax'
            res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
        case 'loss'
            res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
        case 'softmaxloss'
            res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
        case 'svmloss_multiclass'
            res(i).dzdx = vl_nnsvmloss(res(i).x, l.class, res(i+1).dzdx) ;
        case 'relu'
            if ~isempty(res(i).x)
                res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
            else
                % if res(i).x is empty, it has been optimized away, so we use this
                % hack (which works only for ReLU):
                res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx) ;
            end
        case 'sigmoid'
            res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;
        case 'noffset'
            res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
        case 'spnorm'
            res(i).dzdx = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;
        case 'dropout'
            if opts.disableDropout
                res(i).dzdx = res(i+1).dzdx ;
            else
                res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux) ;
            end
        case 'bnorm'
            if ~opts.accumulate
                if isfield(l, 'weights')
                    [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                        vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                        res(i+1).dzdx) ;
                else
                    [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                        vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                        res(i+1).dzdx) ;
                end
            else
                dzdw = cell(1,2) ;
                if isfield(l, 'weights')
                    [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                        vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                        res(i+1).dzdx) ;
                else
                    [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                        vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                        res(i+1).dzdx) ;
                end
                for j=1:2
                    res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
                end
                clear dzdw ;
            end
        case 'pdist'
            res(i).dzdx = vl_nnpdist(res(i).x, l.p, res(i+1).dzdx, ...
                'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
        case 'custom'
            res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    if opts.conserveMemory
        res(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
        wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
end

end
