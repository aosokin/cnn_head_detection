function BB = combine_global(BB, global_det_path, img_w, img_h, varargin)

opts = struct;                   % options of global model
opts.alpha = 0;
opts.scale_range = [1 2 4 8];
opts.stride_proportion = 2;
opts.hm_size = {};
for scl=opts.scale_range
    tile_size = 224/scl;
    stride = tile_size/opts.stride_proportion;
    opts.hm_size{scl} = floor((224-stride)/stride);
end
opts.path_format = '';
opts.platform = 'matconvnet';

opts = vl_argparse(opts, varargin);

%global term
if (opts.alpha > 0)
    output = load(global_det_path, '-mat');
    switch opts.platform
        case 'matconvnet'
            output = output.score(1,:)';
        case 'torch'
            output = output.x(2,:)';
    end
    
    cnt = 0;
    for scl=opts.scale_range
        hm{scl} = reshape(output(cnt+1:cnt+opts.hm_size{scl}^2), ...
            opts.hm_size{scl}, opts.hm_size{scl});
        cnt = cnt+opts.hm_size{scl}^2;
    end
    
    % update score
    %combine with global term
    scale_factor = 224/img_w;
    pad_size = floor((img_w-img_h)/2);
    
    BB_pad = BB;
    BB_pad(:, 2) = BB_pad(:, 2) + pad_size;
    BB_pad = BB_pad*scale_factor;
    BB_pad = convertBb_X1Y1WH_to_X1Y1X2Y2(BB_pad);
    
    for BB_pad_cnt= 1:size(BB_pad, 1)
        max_ov_allscale = -inf;
        max_c_allscale = 0;
        max_d_allscale = 0;
        max_scl = 0;
        
        for scl=opts.scale_range
            tile_size = 224/scl;
            stride = tile_size/opts.stride_proportion;
            %determine which cell the top-left belonging to
            cell_c_tl = floor((BB_pad(BB_pad_cnt, 1)-1)/stride);
            cell_d_tl = floor((BB_pad(BB_pad_cnt, 2)-1)/stride);
            %determine which cell the bot_right belonging to
            cell_c_br = floor((BB_pad(BB_pad_cnt, 3)-1)/stride);
            cell_d_br = floor((BB_pad(BB_pad_cnt, 4)-1)/stride);
            
            if (cell_c_tl <= 0)
                cell_c_tl = 1;
            end
            if (cell_d_tl <= 0)
                cell_d_tl = 1;
            end
            if (cell_c_br <= 0)
                cell_c_br = 1;
            end
            if (cell_d_br <= 0)
                cell_d_br = 1;
            end
            
            max_ov = -inf;
            max_c = 0;
            max_d = 0;
            for c = cell_c_tl:cell_c_br
                for d = cell_d_tl:cell_d_br
                    ov = bbIntersectionOverUnion([(c-1)*stride+1 (d-1)*stride+1 tile_size tile_size], convertBb_X1Y1X2Y2_to_X1Y1WH(BB_pad(BB_pad_cnt, 1:4)));
                    
                    if (ov > max_ov)
                        max_ov = ov;
                        max_c = c;
                        max_d = d;
                        
                    end
                end
            end
            
            if (max_ov > max_ov_allscale)
                max_ov_allscale = max_ov;
                max_scl = scl;
                max_c_allscale = max_c;
                max_d_allscale = max_d;
            end
        end
        BB(BB_pad_cnt,5) = BB(BB_pad_cnt,5)*(1-opts.alpha) + opts.alpha*hm{max_scl}(max_d_allscale, max_c_allscale);
    end
end