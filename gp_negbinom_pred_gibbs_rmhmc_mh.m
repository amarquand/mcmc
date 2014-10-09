function [yhat, yhat_tr]  = gp_negbinom_pred_gibbs_rmhmc_mh(X, tr, te, y, opt)

% Subject and cross-validation parameters
n = size(y,1);
%nte   = size(Xs{1},1);
ntr   = length(tr);
nte   = length(te);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin k-fold cross-validation block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the posteriors
disp('++ Loading posteriors ...');
load([opt.OutputFilename,'f_all'])
load([opt.OutputFilename,'LogTheta_all'])

f_post        = f_all(:,opt.BurnIn:opt.TestInterval:end);
LogTheta_post = LogTheta_all(:,opt.BurnIn:opt.TestInterval:end);

clear f_all LogTheta_all

% Compute predictions
disp('++ Computing predictions ...');
yhat  = zeros(nte, 1);
yhat_tr = zeros(ntr, 1);
n_test_samples = 0;
for i = 1 : length(f_post);
    K   = feval(opt.CovFunc, X, LogTheta_post(1,i));
    Kt  = K(tr,tr);
    Ks  = K(te,tr);
    Kss = K(te,te);
    %InvKt = invert_K(Kt,y(tr));
    ridge   = 1e-1+eye(size(Kt,1));
    InvKt   = inv(Kt+ridge);
    
    % compute predictive mean and variance
    mu = Ks*InvKt*f_post(:,i);
    Sigma = Kss - Ks*InvKt*Ks';
    s2   = diag(Sigma);
    
    % training set
    mutr    = Kt*InvKt*f_post(:,i);
	Sigmatr = Kt - Kt*InvKt*Kt;
    s2tr = diag(Sigmatr);
    
%     yhat_i = zeros(size(Ks,1),1);
%     for s = 1:nte
%         mus = mu(s) + Sigma(s,s).*randn(1000,1);
%         yhat_i(s)  = sum(likelihood_poisson(mus,repmat(y(te(s)),1000,1),opt.PoissScale));
%          exp(f);
           %%ymu = exp(s(g(sig*t'+mu*oN)+lw));
%     end
%     yhat_i = yhat_i ./ 1000;
%     
%     yhat_tr_i = zeros(size(Ks,1),1);
%     for s = 1:ntr
%         mustr = mutr(s) + Sigmatr(s,s).*randn(1000,1);
%         yhat_tr_i(s) = sum(likelihood_poisson(mustr,repmat(y(tr(s)),1000,1),opt.PoissScale));
%     end
%     yhat_tr_i = yhat_tr_i ./ 1000;
      
    yhat_i = exp(mu + s2/2 + LogTheta_post(2,i));
    yhat_tr_i = exp(mutr + s2tr/2 + LogTheta_post(2,i));
    %yhat_i = opt.PoissScale*exp(mu + s2/2 + LogTheta_post(2,i));
    %yhat_tr_i = opt.PoissScale*exp(mutr + s2tr/2 + LogTheta_post(2,i));

    yhat = yhat + yhat_i;
    yhat_tr= yhat_tr + yhat_tr_i;
    
    n_test_samples = n_test_samples  + 1;
end
yhat  = yhat ./ n_test_samples;
yhat_tr = yhat_tr./ n_test_samples;


