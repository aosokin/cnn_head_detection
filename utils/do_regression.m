function BB = do_regression(BB, img_w, img_h, varargin)
% This function is used to do bounding box regression given translation and
% scale parameters
% Input:
%   - BB: [x y w h]
%   - img_w: image width
%   - img_h: image height
%   - opts.param: [dx dy sx sy]
%   - opts.fix_ann: used to crop frame
% Output:
%   - BB: [x y w h]

opts = struct;               % detection regression parameters
opts.param = [0 0 1 1];
opts.fix_ann = struct;       % fix annotation if needed (e.g. casablanca)
opts.fix_ann.x_off = 0;
opts.fix_ann.y_off = 0;
opts.fix_ann.w = inf;
opts.fix_ann.h = inf;
opts.warp = 'none';

opts = vl_argparse(opts, varargin);

switch opts.warp
    case 'square'
        BB(:, 1:4) = extend_square_head(BB(:, 1:4));
    case 'square_UB' % for TVHI
        BB(:, 1:4) = extend_square_head(BB(:, 1:4));
        BB(:, 1:4) = head_extended_bb_square(BB(:, 1:4));
end

% fix annotation
BB(:,1) = BB(:,1)-opts.fix_ann.x_off;
BB(:,2) = BB(:,2)-opts.fix_ann.y_off;
for j=1:size(BB, 1)
    w = BB(j,3);
    h = BB(j,4);
    %regression
    BB(j, 3) = w*opts.param(3);
    BB(j, 4) = h*opts.param(4);
    BB(j, 1) = BB(j, 1) + w*opts.param(1) -(BB(j, 3)-w)/2;
    BB(j, 2) = BB(j, 2) + h*opts.param(2) -(BB(j, 4)-h)/2;
    
    %fix edge
    BB(j,1:4) = bbIntersection(BB(j,1:4), [1 1 opts.fix_ann.w-opts.fix_ann.x_off+1 opts.fix_ann.h-opts.fix_ann.y_off+1]);
    BB(j,1:4) = bbIntersection(BB(j,1:4), [1 1 img_w img_h]);
end

end