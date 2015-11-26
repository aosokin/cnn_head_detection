function newBb = convertBb_X1Y1X2Y2_to_X1Y1WH( bb )
%convertBb_X1Y1X2Y2_to_X1Y1WH converts the bounding boxes from [X1,Y1,X2,Y2] format to [X1,Y1,W,H] format

numBb = size(bb, 1);

newBb = zeros(numBb, 4);

newBb(:, 1) = bb(:, 1);
newBb(:, 2) = bb(:, 2);
newBb(:, 3) = bb(:, 3) - bb(:, 1) + 1;
newBb(:, 4) = bb(:, 4) - bb(:, 2) + 1;

end

