function build_computeMinMarginalsBinaryMex
% build_computeMinMarginalsBinaryMex builds package computeMinMarginalsBinaryMex
%
% Anton Osokin,  12.04.2015

srcFiles = { 'computeMinMarginalsBinaryMex.cpp' };
allFiles = '';
for iFile = 1 : length(srcFiles)
    allFiles = [allFiles, ' ', srcFiles{iFile}];
end

cmdLine = ['mex ', allFiles, ' -output computeMinMarginalsBinaryMex -largeArrayDims '];
eval(cmdLine);




