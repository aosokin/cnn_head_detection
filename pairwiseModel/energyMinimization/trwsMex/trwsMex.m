%trwsMex optimizes MRF energy using TRW-S or BP algorithm (wrapper to Vladimir Kolmogorov's code).
% http://pub.ist.ac.at/~vnk/papers/TRW-S.html
% 
% This version assumes that pairsise potentials can be "decomposed": V_{ij}(k,l) = P(i, j) * M(k, l).
% Here P depends only on variable indeces, M depends only on labels.
% 
% Input examples:
%   trwsMex(U, P)
%   trwsMex(U, P, M)
%   trwsMex(U, P, M, options)
% Output examples:
%   S = trwsMex(U, P, M, options)
%   [S, E] = trwsMex(U, P, M, options)
%   [S, E, LB] = trwsMex(U, P, M, options)
%   [S, E, LB, lbPlot, energyPlot, timePlot] = trwsMex(U, P, M, options)
% 
% INPUT:
% 	U		- unary terms (double[numLabels, numNodes])
% 	P		- matrix of edge coefficients (sparse double[numNodes, numNodes]); only upper triangle is used
% 	M		- matrix of label dependencies (double[numLabels, numLabels]); if M is not specified, Potts is assumed
% 				if you want to set options without M call: trwsMex(U, P, [], options)
%   options	- Stucture that determines method to be used.
% 				Fields:  
% 					method		:	method to use (string: 'trw-s' or 'bp') default: 'trw-s'
% 					maxIter		:	maximum number of iterations (double) default: 100
% 					funcEps		:	If functional change is less than funcEps then stop, TRW-S only (double) default: 1e-2
% 					verbosity	:	verbosity level: 0 - no output; 1 - final output; 2 - full output (double) default: 0
% 					printMinIter:	After printMinIter iterations start printing the lower bound (double) default: 10
% 					printIter	:	and print every printIter iterations (double) default: 5
% 
% OUTPUT: 
% 	S		- labeling that has energy E, vector numNodes * 1 of type double (indices are in [1,...,numLabels])
%   E       - energy of labeling S
% 	LB		- maximum value of lower bound of type double (only for TRW-S method)
%   lbPlot, energyPlot, timePlot - measurements per iteration
% 
% Anton Osokin (firstname.lastname@gmail.com),  24.09.2014
