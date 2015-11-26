function finalImage = showBoundingBoxes(curImage, boundingBoxes, colors)
%showBoundingBoxes paint the candidate bounding boxes on top of the image
%
% finalImage = showBoundingBoxes(curImage, boundingBoxes, colors)
%
% Input: 
%   curImage - image to show
%   boundingBoxes - double[ numBoxes x 4], each line correponds to the bounding box in [x y w h] format
%   colors - cell array containing colors of the boxed, cell can contain char symbols (ymcrgbwk) or vector of length 3.
%
% Output:
%   finalImage - the image with bounding boxes on top

if max(curImage(:)) > 1
    curImage = single(curImage) / 255;
end

numBoxes = size(boundingBoxes, 1);

if ~exist('colors', 'var') || isempty(colors)
    colors = cell(numBoxes, 1);
    colors(:) = {'m'};
elseif ~iscell(colors)
    colorsOld = colors;
    colors = cell(numBoxes, 1);
    colors(:) = {colorsOld};
end
for iColor = 1 : numel(colors)
    if ischar( colors{iColor} )
        colors{iColor} = getColorFromString( colors{iColor} );
    end
end

finalImage = curImage; 

bBWidth = 2; % width of the dounding box in pixels

boundingBoxes = double( boundingBoxes );
for iBox = numBoxes : -1 : 1
    curColor = colors{iBox};
    
    % draw the line segments
    % top
    finalImage( boundingBoxes(iBox, 2) : boundingBoxes(iBox, 2) + bBWidth - 1, ...
        boundingBoxes(iBox, 1) : boundingBoxes(iBox, 1) + boundingBoxes(iBox, 3) - 1, : ) = ...
        repmat( reshape( curColor, [1 1 3] ), bBWidth, boundingBoxes(iBox, 3) );
    
    % bottom
    finalImage( boundingBoxes(iBox, 2) + boundingBoxes(iBox, 4) - bBWidth : boundingBoxes(iBox, 2) + boundingBoxes(iBox, 4) - 1, ...
        boundingBoxes(iBox, 1) : boundingBoxes(iBox, 1) + boundingBoxes(iBox, 3) - 1, : ) = ...
        repmat( reshape( curColor, [1 1 3] ), bBWidth, boundingBoxes(iBox, 3) );
    
    % left
    finalImage( boundingBoxes(iBox, 2) : boundingBoxes(iBox, 2) + boundingBoxes(iBox, 4) - 1, ...
        boundingBoxes(iBox, 1) : boundingBoxes(iBox, 1) + bBWidth - 1, : ) = ...
        repmat( reshape( curColor, [1 1 3] ), boundingBoxes(iBox, 4), bBWidth );
    
    % right
    finalImage( boundingBoxes(iBox, 2) : boundingBoxes(iBox, 2) + boundingBoxes(iBox, 4) - 1, ...
        boundingBoxes(iBox, 1) + boundingBoxes(iBox, 3) - bBWidth : boundingBoxes(iBox, 1) + boundingBoxes(iBox, 3) - 1, : ) = ...
        repmat( reshape( curColor, [1 1 3] ), boundingBoxes(iBox, 4), bBWidth );
    
end

end

function color = getColorFromString( name )
switch name
    case 'y'
        color = [1, 1, 0];
    case 'm'
        color = [1, 0, 1];
    case 'c'
        color = [0, 1, 1];
    case 'r'
        color = [1, 0, 0];
    case 'g'
        color = [0, 1, 0];
    case 'b'
        color = [0, 0, 1];
    case 'w'
        color = [1, 1, 1];
    case 'k'
        color = [0, 0, 0];
    otherwise 
        error('getColorFromString:unknownColorName', ['Color name ', name, ' is not recognized'] );
end
end
