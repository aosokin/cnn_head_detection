function newBb = convertBb_X1Y1WH_to_Y1X1Y2X2( bb )
%convertBb_X1Y1WH_to_Y1X1Y2X2 converts the bounding boxes from [X1,Y1,W,H] format to [Y1,X1,Y2,X2] format

numBb = size(bb, 1);

newBb = zeros(numBb, 4);

newBb(:, 1) = bb(:, 2);
newBb(:, 2) = bb(:, 1);
newBb(:, 3) = bb(:, 2) + bb(:, 4) - 1; % in [y1 x1 y2 x2] format border pixels are included
newBb(:, 4) = bb(:, 1) + bb(:, 3) - 1;

end

