function res = vl_simplenn_pairwiseModel_forwardPass(net, x, trainableLayers, res, varargin)
%vl_simplenn_pairwiseModel_forwardPass performs the forward pass using the prodided CNN

opts = struct;
opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.saveDataForBackwardPass = true;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

doder = opts.saveDataForBackwardPass;

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;

for i=1:n
    l = net.layers{i} ;
    res(i).time = tic ;
    switch l.type
        case 'conv'
            res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, 'pad', l.pad, 'stride', l.stride) ;
        case 'convPtr'
            id = l.index;
            res(i+1).x = vl_nnconv(res(i).x, trainableLayers{id}.weights{1}, trainableLayers{id}.weights{2}, ...
                'pad', trainableLayers{id}.pad, 'stride', trainableLayers{id}.stride) ;
            
        case 'pool'
            res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
        case 'normalize'
            res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
        case 'softmax'
            res(i+1).x = vl_nnsoftmax(res(i).x) ;
        case 'loss'
            res(i+1).x = vl_nnloss(res(i).x, l.class) ;
        case 'softmaxloss'
            res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
        case 'svmloss_multiclass'
            res(i+1).x = vl_nnsvmloss(res(i).x, l.class) ;
        case 'relu'
            res(i+1).x = vl_nnrelu(res(i).x) ;
        case 'sigmoid'
            res(i+1).x = vl_nnsigmoid(res(i).x) ;
        case 'noffset'
            res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
        case 'spnorm'
            res(i+1).x = vl_nnspnorm(res(i).x, l.param) ;
        case 'dropout'
            if opts.disableDropout
                res(i+1).x = res(i).x ;
            elseif opts.freezeDropout
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
            else
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
            end
        case 'bnorm'
            if isfield(l, 'weights')
                res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
            else
                res(i+1).x = vl_nnbnorm(res(i).x, l.filters, l.biases) ;
            end
        case 'pdist'
            res(i+1) = vl_nnpdist(res(i).x, l.p, 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
        case 'custom'
            res(i+1) = l.forward(l, res(i), res(i+1)) ;
        otherwise
            error('Unknown layer type %s', l.type) ;
    end
    % optionally forget intermediate results
    forget = opts.conserveMemory ;
    forget = forget & (~doder || strcmp(l.type, 'relu')) ;
    forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
    forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
    if forget
        res(i).x = [] ;
    end
    if gpuMode & opts.sync
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice) ;
    end
    res(i).time = toc(res(i).time) ;
end

end
