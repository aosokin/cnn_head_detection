function [labels, energy, isOptimal] = minimizeEnergyPairwiseBinary( unaryTerms, pairwiseTerms, varargin )
%minimizeEnergyPairwiseBinary minimizes (approximately) the energy consisting of the unary and the pairwise potentials.
% Several methods are applied: first - QPBO; if the number of unlabelled nodes is small, the exhaustive search, and TRW-S otherwise.

if ~exist('varargin', 'var')
    varargin = {};
end

%% parameters
opts = struct;
opts.energyComputationTolerance = 1e-5;
opts.labelingToCheck = [];
opts.maxBruteForceVars = 20;
% parse input
opts = vl_argparse(opts, varargin);


%% check input
numLabels = 2;
if ~isnumeric(unaryTerms) || ~ismatrix(unaryTerms) || size(unaryTerms, 2) ~= 2
    error('Incorrect format for unaryTerms, has to be numNodes x 2')
end
numNodes = size(unaryTerms, 1);

if ~isnumeric(pairwiseTerms) || ~ismatrix(pairwiseTerms) || size(pairwiseTerms, 2) ~= 6
    error('Incorrect format for pairwiseTerms, has to be numEdges x 6')
end
numEdges = size(pairwiseTerms, 1);

%% Run QPBO to get partially optimal labelling
[lowerBoundQpbo, labels_qpbo] = qpboMex(double(unaryTerms), double(pairwiseTerms));
maskUnlabeled = labels_qpbo < 0;
numUnlabeled = sum(maskUnlabeled);

labels_qpbo = labels_qpbo + 1;
if numUnlabeled == 0
    % everything is labeled
    labels = labels_qpbo;
    energy = lowerBoundQpbo;
    isOptimal = true;
    return;
end
isOptimal = false;

%% Reduce energy to only unlabeled nodes
if numUnlabeled < numNodes
    [ unaryTermsNew, pairwiseTermsNew, energyConstant ] = projectEnergyBinaryPairwise( unaryTerms, pairwiseTerms, labels_qpbo );
else
    unaryTermsNew = unaryTerms;
    pairwiseTermsNew = pairwiseTerms;
    energyConstant = 0;
end

if numUnlabeled < opts.maxBruteForceVars
    %% run brute force
    [ energy, labelsPartial ] = bruteForceBinaryPairwiseMex(double(unaryTermsNew), double(pairwiseTermsNew));
else
    
    %% Run TRW-S
    [ unaryTerms_reparam, pairwiseTerms_reparam, reparametrizationConstant ] = reparameterizeEnergy( unaryTermsNew, pairwiseTermsNew );
    
    numNodesNew = length( unaryTerms_reparam );
    unaryTrws = [zeros(1, numNodesNew); unaryTerms_reparam'];
    
    maskChange = pairwiseTerms_reparam(:, 1) > pairwiseTerms_reparam(:, 2);
    swapVariable = pairwiseTerms_reparam(maskChange, 1);
    pairwiseTerms_reparam(maskChange, 1) = pairwiseTerms_reparam(maskChange, 2);
    pairwiseTerms_reparam(maskChange, 2) = swapVariable;
    
    pairwiseTrws = sparse( pairwiseTerms_reparam(:, 1), pairwiseTerms_reparam(:, 2), pairwiseTerms_reparam(:, 3), numNodesNew, numNodesNew );
    metricTrws = [0 0 ; 0 1];
    
    optionsTrws = struct;
    optionsTrws.verbosity = 0;
    optionsTrws.funcEps = 1e-6;
    optionsTrws.maxIter = 1000;
    [labelsPartial, energy_trws] = trwsMex_time(double(unaryTrws), double(pairwiseTrws), double(metricTrws), optionsTrws);
    energy = energy_trws + reparametrizationConstant;
    
    % energyCheck = computeEnergyBinaryPairwise( unaryTermsNew, pairwiseTermsNew, labels_trws );
    % if abs( energyCheck - energy ) > 1e-5
    %     error('TRW-S energy is computed wrong')
    % end
end


%% check if provided labeling is better
if ~isempty(opts.labelingToCheck)
    labelsOfInterest = opts.labelingToCheck(maskUnlabeled);
    energyCheck = computeEnergyBinaryPairwise( unaryTermsNew, pairwiseTermsNew, labelsOfInterest );
    
    if energyCheck < energy
        energy = energyCheck;
        labelsPartial = labelsOfInterest;
    end
end

%% produce the result
labels = labels_qpbo;
labels(maskUnlabeled) = labelsPartial;
energy = energy + energyConstant;

%% check energy computation
energy_check = computeEnergyBinaryPairwise( unaryTerms, pairwiseTerms, labels );
if abs(energy - energy_check ) > opts.energyComputationTolerance
    warning(['Energy is computed incorrectly, error: ', num2str(abs(energy - energy_check ))]);
end


end

