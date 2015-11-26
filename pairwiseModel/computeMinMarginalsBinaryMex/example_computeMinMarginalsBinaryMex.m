% example of usage of package computeMinMarginalsBinaryMex
%
% Anton Osokin,  12.04.2015

numNodes = 4;
numEdges = 5;

% [Dp(1), Dp(2)] - unary terms
unaryPotentials=[
    0,16;
    0,13;
    20,0;
    4,0
];

% [p, q, Vpq(1, 1), Vpq(1, 2), Vpq(2, 1), Vpq(2, 2)] - pairwise terms
pairwisePotentials=[
    1,2,0,10,4,0;
    1,3,0,12,-1,0;
    3,2,0,9,-1,0;
    2,4,0,14,0,0;
    3,4,0,0,7,0
    ];

[minMarginals, minMarginals_args] = computeMinMarginalsBinaryMex(unaryPotentials, pairwisePotentials);

% % correct answer: 
% minMarginals = [22 29; 22 29; 24 22; 22 25 ];
% minMarginals_args = cat(3, [0 1; 0 1; 0 0; 0 0], [0 1; 0 1; 0 0; 0 0], [1 1; 1 1; 0 1; 1 1], [0 1; 0 1; 0 0; 0 1]);

if ~isequal(minMarginals, [22 29; 22 29; 24 22; 22 25 ])
    warning('Wrong value of the min-marginals!')
end
if ~isequal(minMarginals_args, cat(3, [0 1; 0 1; 0 0; 0 0], [0 1; 0 1; 0 0; 0 0], [1 1; 1 1; 0 1; 1 1], [0 1; 0 1; 0 0; 0 1]) )
    warning('Wrong values of the args of the min-marginals!')
end
