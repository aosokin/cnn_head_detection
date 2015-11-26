function newBb = convertBb_Y1X1Y2X2_to_X1Y1WH( bb )
%convertBb_Y1X1Y2X2_to_X1Y1WH converts the bounding boxes from [Y1,X1,Y2,X2] format to [X1,Y1,W,H] format

numBb = size(bb, 1);

newBb = zeros(numBb, 4);

newBb(:, 1) = bb(:, 2);
newBb(:, 2) = bb(:, 1);
newBb(:, 3) = bb(:, 4) - bb(:, 2) + 1; % in [y1 x1 y2 x2] format border pixels are included
newBb(:, 4) = bb(:, 3) - bb(:, 1) + 1;

end

