function res = bbIntersection(rect1, rect2)
res = zeros(1, 4);
res(1) = max(rect1(1), rect2(1));
res(2) = max(rect1(2), rect2(2));

res(3) = min(rect1(1) + rect1(3), rect2(1) + rect2(3)) - res(1);
res(4) = min(rect1(2) + rect1(4), rect2(2) + rect2(4)) - res(2);

end