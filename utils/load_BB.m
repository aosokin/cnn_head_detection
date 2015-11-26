function bb = load_BB(detector_type, res_path)
    %load bbox+score for each type of detector
    % Input:
    %       - detector_type: 
    %                   'local': willow head local model
    %                   'rcnn': rcnn model
    %                   'rcnn_svm': rcnn model + svm on top
    %                   'pairwise': unary + binary
    %                   'dpm':      DPM face detector
    %       - res_path: path to load bb
    % Output:
    %       - BB = [x y w h]
    bb = [];
    
    res = load(res_path, '-mat');
    
    switch detector_type
        case 'rcnn_svm'
            bb = [bb; res.scores(:,1:5)];
            bb(:, 1:4) = convertBb_X1Y1X2Y2_to_X1Y1WH(bb(:, 1:4));
        case 'pairwise'
            bb = res.BB;
        case 'local'
            bb = res.BB;
        case 'dpm'
            bb = res.ds(:,[1 2 3 4 6]);    
            bb(:, 1:4) = convertBb_X1Y1X2Y2_to_X1Y1WH(bb(:, 1:4));
    end