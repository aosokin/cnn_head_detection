function det = load_det(varargin)
% Load detection results of different local/pairwise models

%% parse parameters
opts = struct;
% model type
opts.fix_ann.x_off = 0;
opts.fix_ann.y_off = 0;
opts.fix_ann.w = inf;
opts.fix_ann.h = inf;                   

opts.regression = struct;               % detection regression parameters
opts.regression.param = [0 0 1 1];
opts.regression.fix_ann = struct;       % fix annotation if needed
opts.regression.fix_ann.x_off = 0;
opts.regression.fix_ann.y_off = 0;
opts.regression.fix_ann.w = inf;
opts.regression.fix_ann.h = inf;                   
opts.regression.warp = 'none';

opts.nms = struct;
opts.nms.nmsIntersectionOverAreaThreshold = 0.3;
opts.nms.numBoundingBoxMax = inf;

opts.regression_before_nms = false;     % do regression before nms

opts.det = struct;
opts.det.modeltype = 'local';
opts.det.thres = -inf;
opts.det.path_format = '';
opts.det.scoretype = 'raw';
opts.det.as_filter = false;             % remove *weird* aspect ratio
opts.det.as_ratio = 1.5;

opts.im_path_format = '';
opts.im_set = [];

opts.viz = struct;                      % visulization
opts.viz.doviz = false;
opts.viz.max_det = 5;

opts.verbose = true;
opts.progress_part = 5;

opts = vl_argparse(opts, varargin);

det = struct;
numimage = length(opts.im_set);

fprintf('Loading detections');
progress_part_num = ceil(numimage/opts.progress_part);

for i = 1:numimage
    if (opts.verbose)
        if (i==numimage)
            fprintf('...100%%\n');
        else
            if ~mod(i,progress_part_num)
                fprintf('...%d%%', i*100/(progress_part_num*opts.progress_part));
            end
        end
    end
    
    if (~iscell(opts.im_set))
        idname = opts.im_set(i);
        det_path = sprintf(opts.det.path_format, idname);
        im_path = sprintf(opts.im_path_format , idname);
        
    else
        idname = opts.im_set{i};
        det_path = sprintf(opts.det.path_format, opts.im_set{i});
        im_path = sprintf(opts.im_path_format , opts.im_set{i});
    end
    det(i).path = det_path;
    det(i).impath = im_path;
    det(i).id = idname;
    
    im = imread(im_path); 
    [img_h, img_w, ~] = size(im);
    
    BB = load_BB(opts.det.modeltype, det_path);
    if (isempty(BB))
        det(i).bb = BB;
        continue;
    end
    
%     if (strcmp(detector,'globalmasked')~=0)
%         scale_factor = 224/size(im,2);
%         pad_size = floor((size(im,2)-size(im,1))/2);
%         BB_unpad = BB/scale_factor;
%         BB_unpad(:, 2) = BB_unpad(:, 2) - pad_size;
%         BB = BB_unpad;
%     end
    
    %thresholding lowscore detections
    BB = BB(BB(:,5)>opts.det.thres, :);
    if (isempty(BB)) det(i).bb = BB; continue; end
    
    %remove *weird* aspect ratio if needed
    if (opts.det.as_filter)
        w = BB(:,3);
        h = BB(:,4);
        as = w./h;
        BB = BB(as < opts.det.as_ratio && as > 1/opts.det.as_ratio, :);
    end
    if (isempty(BB)) det(i).bb = BB; continue; end
    
    if (~opts.regression_before_nms)
        top = selectBoundingBoxesNonMaxSup(BB(:,1:4), BB(:,5), opts.nms);
        BB = BB(top, :);
        BB = do_regression(BB, img_w, img_h, opts.regression);
    else
        BB = do_regression(BB, img_w, img_h, opts.regression);
        top = selectBoundingBoxesNonMaxSup(BB(:,1:4), BB(:,5), opts.nms);
        BB = BB(top, :);
    end
    
    if (opts.viz.doviz)
        fig = figure;
        imshow(im); hold on;
        for z=1:min(opts.viz.max_det, size(BB,1))
            rectangle('position', BB(z, 1:4), 'edgecolor', 'r');
            text(double(BB(z, 1))+5, double(BB(z, 2))+5, num2str(BB(z, 5)), 'color', 'yellow');
        end
        disp('Press any key to continue...');
        pause;
        close(fig);
    end
    
    det(i).bb = BB;
end
end