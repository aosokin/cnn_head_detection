function IoA = bbIntersectionOverArea( candidates, reference )
%bbIntersectionOverArea computes the IoA score between bounding boxes.
% For two bounding boxes the IoU score is the area of the intersection divided by the area of the reference box
%
% IoA = bbIntersectionOverArea( candidates, reference );
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
%   IoA - vector of the scores, double[ numBoundingBoxes, 1 ]

if ~isnumeric( candidates ) || size( candidates, 2 ) ~= 4
    error('bbIntersectionOverArea:badInputCandidates', 'Input <candidates> is of incorrect format');
end
if ~isnumeric( reference ) || size( reference, 2 ) ~= 4 || size( reference, 1 ) ~= 1
    error('bbIntersectionOverArea:badInputReference', 'Input <reference> is of incorrect format');
end
    
area = bbIntersectionArea( candidates, reference );
IoA = area ./ ( reference(3) * reference(4) ); 

end



