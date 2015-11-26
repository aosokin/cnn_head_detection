% example of usage of package trwsMex
%
% Anton Osokin (firstname.lastname@gmail.com), 24.09.2014

% this example runs trwsMex_time on a simple binary energy of 5 variables
% y1 - y2 + y1 * y5 - 10 * y1 * y3 - y3 * y4 + y3 * y5  

dataCost = [0 0 0 0 0; 1 -1 0 0 0];

neighbors = sparse([1; 1; 3; 3], [5; 3; 4; 5], [1; -10; -1; 1], 5, 5);

metric = [0 0; 0 1];

options.maxIter = 100;
options.verbosity = 1;
[labels, energy, LB] = trwsMex(dataCost, neighbors, metric, options);

% % correct answer: 
% energy = -11; 
% labels = [2; 2; 2; 2; 1];

if ~isequal(energy, -11)
    warning('Wrong value of energy!')
end
if ~isequal(labels, [2; 2; 2; 2; 1])
    warning('Wrong value of labels!')
end
