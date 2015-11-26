function newBb = convertBb_X1Y1WH_to_X1Y1X2Y2( bb )
%convertBb_X1Y1WH_to_X1Y1X2Y2 converts the bounding boxes from [X1,Y1,W,H] format to [X1,Y1,X2,Y2] format

numBb = size(bb, 1);

newBb = zeros(numBb, 4);

newBb(:, 1) = bb(:, 1);
newBb(:, 2) = bb(:, 2);
newBb(:, 3) = bb(:, 1) + bb(:, 3) - 1;
newBb(:, 4) = bb(:, 2) + bb(:, 4) - 1;

end

