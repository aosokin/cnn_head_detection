function build_bruteForceBinaryPairwiseMex
% build_bruteForceBinaryPairwiseMex builds package bruteForceBinaryPairwiseMex
%
% Anton Osokin,  03.04.2015

srcFiles = { 'bruteForceBinaryPairwiseMex.cpp' };
allFiles = '';
for iFile = 1 : length(srcFiles)
    allFiles = [allFiles, ' ', srcFiles{iFile}];
end

cmdLine = ['mex ', allFiles, ' -output bruteForceBinaryPairwiseMex -largeArrayDims '];
eval(cmdLine);




