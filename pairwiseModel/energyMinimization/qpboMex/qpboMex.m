% qpboMex - Matlab interface to Vladimir Kolmogorov's implementation of QPBO algorithm downloadable from:
% http://www.cs.ucl.ac.uk/staff/V.Kolmogorov/software.html
% 
% Energy function:
% E(x)   =   \sum_p D_p(x_p)   +   \sum_pq V_pq(x_p,x_q)
% where x_p \in {0, 1},
% Vpq(0,0), Vpq(0, 1), Vpq(1,0), Vpq(1,1) can be arbitrary
% Wrapper is computing weak (not strong!) persistent solution.
% 
% Usage:
% [LB] = qpboMex(unaryTerms, pairwiseTerms);
% [LB, labels] = qpboMex(unaryTerms, pairwiseTerms);
% 	
% Inputs:
% unaryTerms - of type double, array size [numNodes, 2]; the cost of assigning 0, 1 to the corresponding unary term ([Dp(0), Dp(1)])
% pairwiseTerms - of type double, array size [numEdges, 6]; each line corresponds to an edge [p, q, Vpq(0,0), Vpq(0, 1), Vpq(1,0), Vpq(1,1)];
% 				p and q - indecies of vertecies from 1,...,numNodes, p != q;
% 
% Outputs:
% LB - of type double, a single number; lower bound found by QPBO
% labels - of type double, array size [numNodes, 1] of {0, 1, -1}; labeling found by QPBO; -1 means refusal to label the vertex
% 
% Anton Osokin, firstname.lastname@gmail.com, 24.09.2014 
