function  idsNms = selectBoundingBoxesNonMaxSup( boundingBoxes, scores, varargin )
%selectBoundingBoxesNonMaxSup performs the NMS on the candidate bounding boxes
% boxes should be in [X1,Y1,W,H] format
%
% idsNms = selectBoundingBoxesNonMaxSup( boundingBoxes, scores )
%
% Input: 
%   boundingBoxes - double[ numBoxes x 4], each line correponds to the bounding box in [X1,Y1,W,H] format
%   scores - double[numBoxes x 1], scores of the bounding boxes, will be sorted in the decreasing order
% 
% Extra parameters: 
%   nmsIntersectionOverAreaThreshold - IoA threshold used to select boxes
%   numBoundingBoxMax - maximum number of boxes selected by NMS
%
% Output:
%   idsNms - indices of the bounding boxes selected by NMS

if ~exist('varargin', 'var')
    varargin = {};
end
%% parameters
opts = struct;
opts.numBoundingBoxMax = inf;
opts.nmsIntersectionOverAreaThreshold = 0.3;
opts = vl_argparse(opts, varargin);

%% do the job
numBbs = length(scores);
[~, ids] = sort(scores, 'descend');

idsNms = nan( min(opts.numBoundingBoxMax, numel(ids)), 1 );
idsNms(1) = ids(1);

numBbNms = 1;
iBb = 1;
while numBbNms < opts.numBoundingBoxMax && iBb < numBbs
    curIou = inf;
    while max( curIou(:) ) > opts.nmsIntersectionOverAreaThreshold && iBb < numBbs
        iBb = iBb + 1;
        curIou = bbIntersectionOverArea( boundingBoxes( idsNms(1 : numBbNms), : ), boundingBoxes( ids(iBb), : ) );
    end
    
    if max( curIou(:) ) <= opts.nmsIntersectionOverAreaThreshold
        numBbNms = numBbNms + 1;
        idsNms( numBbNms ) = ids( iBb );
    end
end

idsNms = idsNms(1 : numBbNms);

end

