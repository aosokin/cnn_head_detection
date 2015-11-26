function compile_mex( cudaRoot )
%compile_mex compiles all the MEX-functions included in this package

rootDir = fileparts( mfilename( 'fullpath' ) );

% image cropping on a GPU
cd( fullfile(rootDir, 'utils', 'cropRectanglesMex') );
if exist('cudaRoot', 'var')
	build_cropRectanglesMex( cudaRoot );
else
	build_cropRectanglesMex;
end
cd(rootDir);

% computation of the min-marginals
cd( fullfile(rootDir, 'pairwiseModel', 'computeMinMarginalsBinaryMex') );
build_computeMinMarginalsBinaryMex;
cd(rootDir);

% brute force energy minimization
cd( fullfile(rootDir, 'pairwiseModel', 'energyMinimization', 'bruteForceBinaryPairwiseMex') );
build_bruteForceBinaryPairwiseMex;
cd( rootDir );

% QPBO
cd( fullfile(rootDir, 'pairwiseModel', 'energyMinimization', 'qpboMex') );
build_qpboMex;
cd( rootDir );

% TRW-S
cd( fullfile(rootDir, 'pairwiseModel', 'energyMinimization', 'trwsMex') );
build_trwsMex;
cd( rootDir );


end
