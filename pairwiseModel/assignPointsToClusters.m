function clusterIds = assignPointsToClusters( clusterCenters, features )
%assignPointsToClusters assigns point to clusters using L2-distance to the cluster centers
%
% clusterIds = assignPointsToClusters( clusterCenters, features )
%
% Input:
%   clusterCenters - centers, double[ numClusters x numFeatures ]
%   features - points, double[ numPoints x numFeatures ]
%
% Output: 
%   clusterIds - each point is assign to a cluster, 1-based indexing, double[numPoints x 1]

dist = -2 * features * clusterCenters';

cL2 = sum(clusterCenters .^ 2, 2);
fL2 = sum(features .^ 2, 2);

dist = bsxfun(@plus, dist, fL2);
dist = bsxfun(@plus, dist, cL2');

[~, clusterIds] = min( dist, [], 2 );

end

