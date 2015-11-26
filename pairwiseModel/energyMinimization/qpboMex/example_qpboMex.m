% example of usage of package qpboMex
%
% Anton Osokin (firstname.lastname@gmail.com),  24.09.2014

nNodes=4;

% [Dp(0), Dp(1)] - unary terms
terminalWeights=[
    0,16;
    0,13;
    20,0;
    4,0
];

% [p, q, Vpq(0, 0), Vpq(0, 1), Vpq(1,0), Vpq(1, 1)] - pairwise terms
edgeWeights=[
    1,2,0,10,4,0;
    1,3,0,12,-1,0;
    2,3,0,-1,9,0;
    2,4,0,14,0,0;
    3,4,0,0,7,0
    ];

[lowerBound, labels] = qpboMex(terminalWeights, edgeWeights);

% % correct answer: 
% lowerBound = 22; 
% labels = [0; 0; 1; 0];

if ~isequal(lowerBound, 22)
    warning('Wrong value of lowerBound!')
end
if ~isequal(labels, [0; 0; 1; 0])
    warning('Wrong value of labels!')
end
