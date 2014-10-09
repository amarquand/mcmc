clc
clear

% Subject parameters
root_dir    = '/home/andre/cns/svmdata/PD_MSA_PSP/';
k_dir       = [root_dir,'/kernels69_4class_MSACP_standardised_allvox/'];
working_dir = '/home/andre/tmp/mcmc_test/';
output_dir  = [working_dir];

n_classes  = 4;
n_gps      = n_classes;
n_subjects = 50;
n_sources  = 2;
cv_mode    = 'loo2';
fold       = 1;
modalities = {'s1mnet','s2mnet'};

mkdir(output_dir);
output_prefix = [output_dir,'cv_fold_',num2str(fold),'_'];

%%%%%%%%%%%%%%%%%%%%%%%
% Configure parameters
%%%%%%%%%%%%%%%%%%%%%%%
opt.nGibbsIter     = 10000;    
opt.BurnIn         = 1000;
opt.PriorParam     = [2,2];
opt.OptimiseTheta  = true;
opt.CovFunc        = 'covfunc_sum'; %'covfunc_wsum';
opt.OutputFilename = output_prefix; % leave blank for no output
opt.UseRMHMC       = false;   
opt.UseGMassForHMC = true; % Only used if UseRMHMC = false
opt.X0_RMHMC       = zeros(n_gps*n_subjects,1);
opt.X0_MH          = zeros(n_gps*n_sources,1);

% set parameters for f ...
opt.rmhmc.NumOfIterations    = 1; 
opt.rmhmc.StepSize           = 0.5;
opt.rmhmc.NumOfLeapFrogSteps = 10;
opt.rmhmc.NumOfNewtonSteps   = 4;

% ... and theta
%opt.mh.UseGMassForHMC     = true; % Only used if UseRMHMC = false
%opt.mh.NumOfIterations    = 1; 
opt.mh.StepSize           = 0.2; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Input Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load labels
load([k_dir,'labels']);

% set a default for the classes used (subset of total)
if isfield(opt,'classes')
    Y = Y(:,opt.classes);
end
sidx = sum(Y,2) ~= 0;
Y = Y(sidx,:);
% load the kernels
for k = 1:length(modalities)
    %k
    load([k_dir,'kernel_',modalities{k}]);
    
    %K_all{k} = K;
    %K_all{k} = normalize_kernel(K);
    %K_all{k} = prt_normalise_kernel(K(sidx,sidx));
    K_all{k} = prt_scale_kernel('mdiag',K(sidx,sidx));
    
    clear K;
end

% Configure training and test indices for cross-validation
[train,test] = configure_cv(Y,fold,cv_mode);

Kt_all = {}; Ks_all = {};
for k = 1:length(K_all)
   Kt_all{k} = K_all{k}(train,train); 
   Ks_all{k} = K_all{k}(train,test); 
end
Yt = Y(train,:);

[stats] = gpc_gibbs_rmhmc_mh(K_all, Y,opt);