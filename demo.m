% Demo code
% Tuan-Hung Vu, Anton Osokin, Ivan Laptev, Context-aware CNNs for person head detection, ICCV 2015

%% Setup
setup;

%% Training ===============================
% %% To rerun our training and evaluation uncomment the following lines, specify matconvnetPath and cudaRoot to the root folder of Matconvnet and CUDA
% matconvnetPath = '~/local/software/matlab_toolboxes/matconvnet-1.0-beta12';
% setup( matconvnetPath );
%
% cudaRoot = '/usr/cuda-7.0' ;
% compile_mex(cudaRoot);
%
% %% Train/evaluate local model
% run_training_localModel;
% run_computeScores_localModel;
% 
% %% Train/evaluate pairwise model
% run_training_pairwiseModel;
% run_computeScores_pairwiseModel;
%
% %% Train/evaluate global model
% run_training_globalModel;
% run_computeScores_globalModel;

%% Evaluation ===============================
% Following code will evaluate results of different models on
% HollywoodHeads test set.
% List of evaluated models
%   - local
%   - pairwise
%   - local + pairwise + global
%   - RCNN
%   - DPM

VOCinit_HH; % prepare VOC options
RES_ROOT = fullfile(VOCopts.resdir, 'res');
if ~exist(RES_ROOT, 'dir')
    mkdir(RES_ROOT)
end

%load testset
im_format = VOCopts.imgpath;
test_set = readLines(sprintf(VOCopts.imgsetpath, 'test'));

%pre-saved detection path format
LOCAl_SAVE_DET_FORMAT = 'results/HollywoodHeads/local/dets/%s.mat';
GLOBAL_SAVE_DET_FORMAT = 'results/HollywoodHeads/global/dets/%s.mat';
PAIRWISE_SAVE_DET_FORMAT = 'results/HollywoodHeads/pairwise/dets/%s.mat';
RCNN_SAVE_DET_FORMAT = 'results/HollywoodHeads/rcnn/dets/%s.mat';
DPM_SAVE_DET_FORMAT = 'results/HollywoodHeads/dpm/dets/%s.mat';

%% Local model
modelname = 'Local_model';
disp(['> Evaluating model: ' modelname]);

saverespath = fullfile(RES_ROOT, [modelname '.mat']);
if ~exist(saverespath, 'file')
    opts = struct;
    opts.det.modeltype = 'local';
    opts.det.scoretype = 'raw';
    opts.det.path_format = LOCAl_SAVE_DET_FORMAT;
    opts.im_path_format = im_format;
    opts.im_set = test_set;
    opts.viz.doviz = false;
    
    %regressed version
    opts.regression.param = [0.000500595642863 -0.00649301373331 0.975682458365 0.982287603066];
    det = load_det(opts);
    [rec, prec, ap] = evaluate_detection_HH(det, modelname, VOCopts, 'head');
    
    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% Pairwise model
modelname = 'Pairwise_model';
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
    opts.regression.param = [0.000500595642863 -0.00649301373331 0.975682458365 0.982287603066];
    det = load_det(opts);
    [rec, prec, ap] = evaluate_detection_HH(det, modelname, VOCopts, 'head');
    
    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% Local + global + pairwise model
modelname = 'Final_model';
disp(['> Evaluating model: ' modelname]);
saverespath = fullfile(RES_ROOT, [modelname '.mat']);

if ~exist(saverespath, 'file')
    opts = struct;
    opts.local_res_path = fullfile(RES_ROOT, 'Local_model.mat');
    opts.pairwise_res_path = fullfile(RES_ROOT, 'Pairwise_model.mat');
    opts.im_path_format = im_format;
    opts.im_set = test_set;
    opts.regression.param = [0.000500595642863 -0.00649301373331 0.975682458365 0.982287603066];
    
    opts.ialpha = 3;
    opts.ibias = 7;
    
    opts.global.alpha = 0.30;
    opts.global.path_format = GLOBAL_SAVE_DET_FORMAT;
    opts.global.platform = 'matconvnet';
    
    det = load_det_local_pairwise_global(opts);
    [rec, prec, ap] = evaluate_detection_HH(det, modelname, VOCopts, 'head');
    
    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% RCNN
modelname = 'RCNN';
disp(['> Evaluating model: ' modelname]);
saverespath = fullfile(RES_ROOT, [modelname '.mat']);
if (exist(saverespath, 'file') == 0)
    opts = struct;
    opts.regression.param = [-0.00239252348267 -0.017133841922 0.975428819577 0.958512960492];
    opts.det.modeltype = 'rcnn_svm';
    opts.det.scoretype = 'raw';
    opts.det.path_format = RCNN_SAVE_DET_FORMAT;
    opts.im_path_format = im_format;
    opts.im_set = test_set;
    
    det = load_det(opts);
    [rec, prec, ap] = evaluate_detection_HH(det, modelname, VOCopts, 'head');
    
    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end
%% DPM
modelname = 'DPM';
disp(['> Evaluating model: ' modelname]);
saverespath = fullfile(RES_ROOT, [modelname '.mat']);
if (exist(saverespath, 'file') == 0)
    opts = struct;
    opts.regression.param = [-0.0110632673917 -0.154468706569 1.17615081441 1.32509178499];
    opts.det.modeltype = 'dpm';
    opts.det.scoretype = 'raw';
    opts.det.path_format = DPM_SAVE_DET_FORMAT;
    opts.im_path_format = im_format;
    opts.im_set = test_set;
    
    det = load_det(opts);
    [rec, prec, ap] = evaluate_detection_HH(det, modelname, VOCopts, 'head');
    
    save(saverespath, 'rec', 'prec', 'ap', 'det', 'opts', 'VOCopts');
else
    load(saverespath);
end

%% Draw AP curve
%list of visualized methods and corresponding curve-colors
list_model = {'DPM', 'RCNN', 'Local_model', 'Final_model'};
color_hex = {'ffbd0d', '1e72ef', '019c59', 'dd4f3b'};

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
