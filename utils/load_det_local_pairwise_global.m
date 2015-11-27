function det = load_det_local_pairwise_global(varargin)
% Combine detection results of local, pairwise and global models

opts = struct;
opts.alpha_pairwise_range = 0.9:-0.1:0.1;
opts.bias_range = 10:-1:-10;
opts.local_res_path = '';
opts.pairwise_res_path = '';

opts.regression.param = [0 0 1 1];
opts.regression.fix_ann = struct;       % fix annotation if needed
opts.regression.fix_ann.x_off = 0;
opts.regression.fix_ann.y_off = 0;
opts.regression.fix_ann.w = inf;
opts.regression.fix_ann.h = inf;                   
opts.regression.warp = 'none';

opts.global = struct;                   % options of global model
opts.global.alpha = 0;
opts.global.scale_range = [1 2 4 8];
opts.global.stride_proportion = 2;
opts.global.hm_size = {};
for scl=opts.global.scale_range
    tile_size = 224/scl;
    stride = tile_size/opts.global.stride_proportion;
    opts.global.hm_size{scl} = floor((224-stride)/stride);
end
opts.global.path_format = '';
opts.global.platform = 'matconvnet';

opts.ialpha = 1;
opts.ibias = 1;

opts.im_path_format = '';
opts.im_set = [];

opts.verbose = true;
opts.progress_part = 5;

opts = vl_argparse(opts, varargin);

%% Load local detections
load(opts.local_res_path , 'det');
det_lcl = det;
%% load pairwise detections
load(opts.pairwise_res_path , 'det');
det_pw = det;
%% Combined detections
alpha_pairwise = opts.alpha_pairwise_range(opts.ialpha);
bias = opts.bias_range(opts.ibias);

DET = [];
N_img = length(det_lcl);

fprintf('Combining detections\n');
progress_part_num = ceil(N_img/opts.progress_part);

for i=1:N_img
    if (opts.verbose)
        if (i==N_img)
            fprintf('...100%%\n');
        else
            if ~mod(i,progress_part_num)
                fprintf('...%d%%', i*100/(progress_part_num*opts.progress_part));
            end
        end
    end
    d1 = det_lcl(i).bb(:,1:5);
    d2 = det_pw(i).bb(:,1:5);
    
    dist = pdist2(d1(:,1:4), d2(:,1:4));
    [~,m_i] = min(dist);
    
    d_c = zeros(size(d1));
    d_c(1:size(d2,1), 1:4) = d1(m_i, 1:4);
    d_c(1:size(d2,1), 5) =  d1(m_i, 5)*(1-alpha_pairwise) + d2(:,5)*alpha_pairwise + bias;
    
    d_c(size(d2,1)+1:size(d1,1),:) = d1(setdiff(1:size(d1,1), m_i), :);
    
    det(i) = det_lcl(i);
    det(i).bb = d_c;

    if (~iscell(opts.im_set))
        global_det_path = sprintf(opts.global.path_format, idname);
        im_path = sprintf(opts.im_path_format , idname);
    else
        global_det_path = sprintf(opts.global.path_format, opts.im_set{i});
        im_path = sprintf(opts.im_path_format , opts.im_set{i});
    end
    im = imread(im_path); 
    [img_h, img_w, ~] = size(im);
    
    det(i).bb(:,1:5) = combine_global(det(i).bb(:,1:5), global_det_path, img_w, img_h, opts.global);
    det(i).bb(:,1:5) = do_regression(det(i).bb(:,1:5), img_w, img_h, opts.regression);
end


end