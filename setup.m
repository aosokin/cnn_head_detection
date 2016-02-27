function setup( matconvnetPath )
%setup adds all the paths required by this package
%
% setup( matconvnetPath )
%
% Input:
%   matconvnetPath - path to the root of MatConvNet

rootDir = fileparts( mfilename( 'fullpath' ) );

% setup MatConvNet
if exist('vl_setupnn.m', 'file')
    vl_setupnn;
else
    if ~exist( 'matconvnetPath', 'var' )
        warning('MatConvNet path is not provided. Not all functionality is available. Run setup( matconvnetPath ) where matconvnetPath is the path to the MatConvNet installation.')
    else
        run( fullfile(matconvnetPath, 'matlab', 'vl_setupnn.m') );
    end
end

% helper functions
addpath( fullfile(rootDir, 'utils') );
addpath( fullfile(rootDir, 'utils', 'cropRectanglesMex') );
addpath( fullfile(rootDir, 'utils', 'VOCcode') );
addpath( fullfile(rootDir, 'utils', 'HollywoodHeads') );
addpath( fullfile(rootDir, 'utils', 'Casablanca') );


% code for the local model
addpath( fullfile(rootDir, 'localModel') );

% code for the pairwise model
addpath( fullfile(rootDir, 'pairwiseModel') );
addpath( fullfile(rootDir, 'pairwiseModel', 'computeMinMarginalsBinaryMex' ) );
addpath( fullfile(rootDir, 'pairwiseModel', 'energyMinimization' ) );
addpath( fullfile(rootDir, 'pairwiseModel', 'energyMinimization', 'bruteForceBinaryPairwiseMex' ) );
addpath( fullfile(rootDir, 'pairwiseModel', 'energyMinimization', 'qpboMex' ) );
addpath( fullfile(rootDir, 'pairwiseModel', 'energyMinimization', 'trwsMex' ) );

% code for the global model
addpath( fullfile(rootDir, 'globalModel') );
end
