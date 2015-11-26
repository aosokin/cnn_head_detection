function IoU = bbIntersectionOverUnion( candidates, reference )
%bbIntersectionOverUnion computes the IoU score between bounding boxes.
% For two bounding boxes the IoU score is the area of the intersection divided by the area of the union
%
% IoU = bbIntersectionOverUnion( candidates, reference );
%
% Input:
%   candidates - the candidate bounding boxes, double[ numBoundingBoxes x 4]
%   reference - the reference bounding box, double[ 1 x 4]
%
%   Format for the bounding box representation: [x y w h]
%       (x, y) - position of the upper left corner
%       origin (0, 0) is the upper left corner of the image
%
% Output: 
%   IoU - vector of the scores, double[ numBoundingBoxes, 1 ]

if ~isnumeric( candidates ) || size( candidates, 2 ) ~= 4
    error('bbIntersectionOverUnion:badInputCandidates', 'Input <candidates> is of incorrect format');
end
if ~isnumeric( reference ) || size( reference, 2 ) ~= 4 || size( reference, 1 ) ~= 1
    error('bbIntersectionOverUnion:badInputReference', 'Input <reference> is of incorrect format');
end
    
area = bbIntersectionArea( candidates, reference );
IoU = area ./ (candidates(:,3) .* candidates(:,4) + reference(3) * reference(4) - area);

end
