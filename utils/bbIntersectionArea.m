function area = bbIntersectionArea( candidates, reference )
%bbIntersectionArea computes the area of the intersection of the bounding boxes
%
% area = bbIntersectionArea( candidates, reference );
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
%   area - vector of the intersection areas between all the boxes in candidates with the box in reference, double[ numBoundingBoxes, 1 ]

if ~isnumeric( candidates ) || size( candidates, 2 ) ~= 4
    error('bbIntersetionArea:badInputCandidates', 'Input <candidates> is of incorrect format');
end
if ~isnumeric( reference ) || size( reference, 2 ) ~= 4 || size( reference, 1 ) ~= 1
    error('bbIntersetionArea:badInputReference', 'Input <reference> is of incorrect format');
end

leftBound = max( candidates(:,1), reference(1) );
rightBound = min( candidates(:,1) + candidates(:,3), reference(1) + reference(3) );

lowerBound = min( candidates(:,2) + candidates(:,4), reference(2) + reference(4) );
upperBound = max( candidates(:,2), reference(2) );

area = abs((lowerBound - upperBound) .* (rightBound - leftBound));

area( leftBound >= rightBound | lowerBound <= upperBound ) = 0;

end
