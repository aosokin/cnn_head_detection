This software implements the MATLAB mex-wrapper for QPBO energy minimization algorithm by V.Kolmogorov:
http://pub.ist.ac.at/~vnk/software/QPBO-v1.32.src.zip

Anton Osokin, (firstname.lastname@gmail.com)
24.09.2014
https://github.com/aosokin/qpboMex

Please cite the following paper in any resulting publication:

Vladimir Kolmogorov and Cartsen Rother.
Minimizing non-submodular functions with graph cuts - a review.
In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 29(7):1274-1279, July 2007


PACKAGE
-----------------------------

./qpboMex.cpp - the C++ code of the wrapper

./build_qpboMex.m - function to build the wrapper

./qpboMex.m - the description of the implemented function

./example_qpboMex.m - the example of usage

./QPBO-v1.32.src - C++ code by Vladimir Kolmogorov (the code is used as is)
http://pub.ist.ac.at/~vnk/software/QPBO-v1.32.src.zip

./qpboMex.mexw64 - Win x64 binary file for the mex-function compiled using MATLAB R2014a + MSVC 2012

./qpboMex.mexa64 - Linux x64 binary file for the mex-function compiled using MATLAB R2012a + gcc-4.4

USING THE CODE
-----------------------------

0) Install MATLAB and one of the supported compilers

1) Run build_qpboMex.m

2) Run example_qpboMex.m to check if the code works

The code was tested under 
- Win7-x64 using MATLAB R2014a and MSVC 2012;
- ubuntu-12.04-x64 using MATLAB R2012a and gcc-4.4

OTHER PACKAGES
-----------------------------

* BK max-flow/min-cut algorithm:
https://github.com/aosokin/graphCutMex_BoykovKolmogorov

BK-algorithm would be the most standard one to solve the graph cut problems.

* BK max-flow/min-cut algorithm with interface supporting dynamic graph cuts:
https://github.com/aosokin/graphCutDynamicMex_BoykovKolmogorov

If you need to solve many similar graph cut problems in a row consider using dynamic graph cuts.

* IBFS max-flow/min-cut algorithm: https://github.com/aosokin/graphCutMex_IBFS

The IBFS algorithm has polynomial time runtime guarantees. The BK does not.
In my experience BK works faster for graphs built for standard 4(8)-connected grid MRFs.
If the graph becomes more complicated (especially hierarchical) consider trying IBFS instead.
