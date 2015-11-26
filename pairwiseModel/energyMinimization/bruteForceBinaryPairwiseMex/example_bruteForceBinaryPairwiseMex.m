% example of usage of package bruteForceBinaryPairwiseMex
%
% Anton Osokin,  03.04.2015

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

[energy, labels] = bruteForceBinaryPairwiseMex(unaryPotentials, pairwisePotentials);

% % correct answer: 
% energy = 22;
% labels = [1; 1; 2; 1];

if ~isequal(energy, 22)
    warning('Wrong value of the energy!')
end
if ~isequal(labels, [1; 1; 2; 1])
    warning('Wrong values of the labels!')
end
