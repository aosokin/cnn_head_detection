% Demo code for Casablanca dataset
% Tuan-Hung Vu, Anton Osokin, Ivan Laptev, Context-aware CNNs for person head detection, ICCV 2015

%% Setup
setup;

% %% Compute the scores
% % %% To recompute the scores uncomment the follwoing lines, specify matconvnetPath and cudaRoot to the root folder of Matconvnet and CUDA
% matconvnetPath = '~/local/software/matlab_toolboxes/matconvnet-1.0-beta12';
% setup( matconvnetPath );
%
% cudaRoot = '/usr/cuda-7.0' ;
% compile_mex(cudaRoot);
%
% run_computeScores_localModel_Casablanca;
% run_computeScores_globalModel_Casablanca;
% run_computeScores_pairwiseModel_Casablanca;

%% Evaluation ===============================
% Following code will evaluate results of different models on
% Casablanca test set.
% List of evaluated models
%   - local
%   - pairwise
%   - local + pairwise + global
%   - RCNN
%   - DPM
%   - VJ-CRF

VOCinit_Casablanca; % prepare VOC options
RES_ROOT = fullfile(VOCopts.resdir, 'res');
if ~exist(RES_ROOT, 'dir')
    mkdir(RES_ROOT)
end

%load testset
im_format = VOCopts.imgpath;
test_set = readLines(sprintf(VOCopts.imgsetpath, 'test'));

%pre-saved detection path format
LOCAl_SAVE_DET_FORMAT = 'results/Casablanca/local/dets/%s.mat';
GLOBAL_SAVE_DET_FORMAT = 'results/Casablanca/global/dets/%s.mat';
PAIRWISE_SAVE_DET_FORMAT = 'results/Casablanca/pairwise/dets/%s.mat';
RCNN_SAVE_DET_FORMAT = 'results/Casablanca/rcnn/dets/%s.mat';
DPM_SAVE_DET_FORMAT = 'results/Casablanca/dpm/dets/%s.mat';
VJCRF_SAVE_DET_FORMAT = 'results/Casablanca/vjcrf/dets/%s.mat';

%% Local model
modelname = 'Local';
disp(['> Evaluating model: ' modelname]);

saverespath = fullfile(RES_ROOT, [modelname '.mat']);
if ~exist(saverespath, 'file')
    opts = struct;
    opts.det.modeltype = 'local';
    opts.det.scoretype = 'raw';
    opts.det.path_format = LOCAl_SAVE_DET_FORMAT;
    opts.im_path_format = im_format;
    opts.im_set = test_set;
    
    %regressed version
    opts.regression.param = [0.00413716611725 0.0363641782673 0.835004792794 0.849700154144];
    opts.regression.fix_ann.x_off = 19;
    opts.regression.fix_ann.y_off = 7;
    opts.regression.fix_ann.w = 976;
    opts.regression.fix_ann.h = 720;
    
    det = load_det(opts);
    [rec, prec, ap] = evaluate_detection_Casablanca(det, modelname, VOCopts, 'head');
    
    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% Pairwise model
modelname = 'Pairwise';
disp(['> Evaluating model: ' modelname]);

saverespath = fullfile(RES_ROOT, [modelname '.mat']);
if ~exist(saverespath, 'file')
    opts = struct;
    
    opts.det.modeltype = 'pairwise';
    opts.det.scoretype = 'raw';
    opts.det.path_format = PAIRWISE_SAVE_DET_FORMAT;
    opts.im_path_format = im_format;
    opts.im_set = test_set;
    
    %regressed version
    opts.regression.param = [0.00413716611725 0.0363641782673 0.835004792794 0.849700154144];
    opts.regression.fix_ann.x_off = 19;
    opts.regression.fix_ann.y_off = 7;
    opts.regression.fix_ann.w = 976;
    opts.regression.fix_ann.h = 720;
    
    det = load_det(opts);
    [rec, prec, ap] = evaluate_detection_Casablanca(det, modelname, VOCopts, 'head');
    
    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% Local + global + pairwise model
modelname = 'Local+Global+Pairwise';
disp(['> Evaluating model: ' modelname]);

saverespath = fullfile(RES_ROOT, [modelname '.mat']);
if ~exist(saverespath, 'file')
    %load non-regressed local detections
    local_model_nonreg_save_path = fullfile(RES_ROOT, 'Local_model_nonreg.mat');
    if ~exist(local_model_nonreg_save_path, 'file')
        opts = struct;
        opts.im_path_format = im_format;
        opts.im_set = test_set;
        opts.det.scoretype = 'raw';
        opts.regression.param =  [0 0 1 1];
        opts.regression.fix_ann.x_off = 19;
        opts.regression.fix_ann.y_off = 7;
        opts.regression.fix_ann.w = 976;
        opts.regression.fix_ann.h = 720;
        opts.det.modeltype = 'local';
        opts.det.path_format = LOCAl_SAVE_DET_FORMAT;
        det = load_det(opts);
        save(local_model_nonreg_save_path, 'det');
        clear det;
    end
    
    %load non-regressed pairwise detections
    pairwise_model_nonreg_save_path = fullfile(RES_ROOT, 'Pairwise_model_nonreg.mat');
    if ~exist(pairwise_model_nonreg_save_path, 'file')
        opts = struct;
        opts.im_path_format = im_format;
        opts.im_set = test_set;
        opts.det.scoretype = 'raw';
        opts.regression.param =  [0 0 1 1];
        opts.regression.fix_ann.x_off = 19;
        opts.regression.fix_ann.y_off = 7;
        opts.regression.fix_ann.w = 976;
        opts.regression.fix_ann.h = 720;
        opts.det.modeltype = 'pairwise';
        opts.det.path_format = PAIRWISE_SAVE_DET_FORMAT;
        det = load_det(opts);
        save(pairwise_model_nonreg_save_path, 'det');
        clear det;
    end
    
    
    %combined detections
    opts = struct;
    opts.local_res_path = local_model_nonreg_save_path;
    opts.pairwise_res_path = pairwise_model_nonreg_save_path;
    opts.im_path_format = im_format;
    opts.im_set = test_set;
    opts.regression.param = [0.00413716611725 0.0363641782673 0.835004792794 0.849700154144];

    
    opts.ialpha = 8;
    opts.ibias = 10;
    
    opts.global.alpha = 0.21;
    opts.global.path_format = GLOBAL_SAVE_DET_FORMAT;
    opts.global.platform = 'matconvnet';
    
    det = load_det_local_pairwise_global(opts);
    [rec, prec, ap] = evaluate_detection_Casablanca(det, modelname, VOCopts, 'head');
    
    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% R-CNN
modelname = 'R-CNN';
disp(['> Evaluating model: ' modelname]);
saverespath = fullfile(RES_ROOT, [modelname '.mat']);
if ~exist(saverespath, 'file')
    opts = struct;

    opts.regression.param = [0.00673045220903 0.030561945979 0.852650858301 0.832706413259];
    opts.regression.fix_ann.x_off = 19;
    opts.regression.fix_ann.y_off = 7;
    opts.regression.fix_ann.w = 976;
    opts.regression.fix_ann.h = 720;

    opts.det.modeltype = 'rcnn_svm';
    opts.det.scoretype = 'raw';
    opts.det.path_format = RCNN_SAVE_DET_FORMAT;
    opts.im_path_format = im_format;
    opts.im_set = test_set;

    det = load_det(opts);
    [rec, prec, ap] = evaluate_detection_Casablanca(det, modelname, VOCopts, 'head');

    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% DPM
modelname = 'DPM Face';
disp(['> Evaluating model: ' modelname]);
saverespath = fullfile(RES_ROOT, [modelname '.mat']);
if ~exist(saverespath, 'file')
    opts = struct;

    opts.regression.param = [-0.00806017230456 -0.0757485649849 0.895993064669 1.0942101767];
    opts.regression.fix_ann.x_off = 19;
    opts.regression.fix_ann.y_off = 7;
    opts.regression.fix_ann.w = 976;
    opts.regression.fix_ann.h = 720;

    opts.det.modeltype = 'dpm';
    opts.det.scoretype = 'raw';
    opts.det.path_format = DPM_SAVE_DET_FORMAT;
    opts.im_path_format = im_format;
    opts.im_set = test_set;

    det = load_det(opts);
    [rec, prec, ap] = evaluate_detection_Casablanca(det, modelname, VOCopts, 'head');

    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% VJ-CRF
modelname = 'VJ-CRF';
disp(['> Evaluating model: ' modelname]);

saverespath = fullfile(RES_ROOT, [modelname '.mat']);
if ~exist(saverespath, 'file')
    opts = struct;
    opts.det.modeltype = 'local';
    opts.det.scoretype = 'raw';
    opts.det.path_format = VJCRF_SAVE_DET_FORMAT;
    opts.im_path_format = im_format;
    opts.im_set = test_set;
    opts.nms.nmsIntersectionOverAreaThreshold = inf; %no nms

    det = load_det(opts);
    [rec, prec, ap] = evaluate_detection_Casablanca(det, modelname, VOCopts, 'head');

    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% Draw AP curve
%list of visualized methods and corresponding curve-colors
list_model = {'VJ-CRF', 'DPM Face', 'R-CNN', 'Local', 'Local+Global+Pairwise'};
color_hex = {'8b008b', 'ffbd0d', '1e72ef', '019c59', 'dd4f3b'};

%draw curves
fig = figure('PaperPositionMode','auto'); hold on;
fontsize = 43;
linewidth =  6;
str_legend = [];
for i=1:length(list_model)
    saverespath = fullfile(RES_ROOT, [list_model{i} '.mat']);
    if exist(saverespath, 'file')
        load(saverespath);
        plot(rec,prec,'-', 'color', hex2rgb(color_hex{i}),'LineWidth',linewidth);
        str_legend{end+1} = sprintf('%s (%.1f%%)', list_model{i}, ap*100);
    end
end
grid;
set(gca, 'FontSize', fontsize);
set(gca, 'YLim', [0 1]);
set(gca, 'XLim', [0 1]);
xlabel('Recall', 'fontsize', fontsize);
ylabel('Precision', 'fontsize', fontsize);
l = legend(str_legend);
set(l, 'Interpreter', 'none');
set(gcf, 'Position', [0 0 1920 1024]);
