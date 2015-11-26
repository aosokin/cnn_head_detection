function build_trwsMex
% build_trwsMex builds package trwsMex
%
% Anton Osokin (firstname.lastname@gmail.com), 24.09.2014

mex trwsMex.cpp src/ordering.cpp src/MRFEnergy.cpp src/treeProbabilities.cpp src/minimize.cpp -output trwsMex -largeArrayDims
