function [stats] = blr_gibbs(X, Y, opt)

% Sample a Full Bayesian linear regression model, where the full generative
% model is below, following the notation of Bishop 2006:
%
%   p(Y|X,W,beta) p(W|alpha) p(beta|a_n,b_n) prod( p(alpha_i|a_a, b_a) )
%
% where:
%   - X is an NxD matrix of covariates
%   - Y is an NxS matrix of targets (S = number of subjects)
%   - W is an DxS are the weights
%   - beta is the noise precision 
%   - alpha_i are ARD parameters for each variable. 
%
% The sampler will successively sample the conditional posteriors for  W, 
% alpha and beta from the Markov chain.
%
% Written by A. Marquand

% Specify functions needed for the matlab compiler
%# function sp_infGrid
%# function sp_covMTL

% seed random numbers
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

% housekeeping parameters
write_interval  = round(opt.nGibbsIter/10);
update_interval = 10; 

% make sure y is a vector
y     = Y(:); y(isnan(y)) = 0;

tic; % start timer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N,D] = size(X);  
S = size(Y,2);

% % polynomial basis expansion for trend coefficients
% Phi = zeros(size(X,1),size(X,2)*opt.DimPoly); colid = 1:size(X,2);
% for d = 1:opt.DimPoly
%     Phi(:,colid) = X.^d; colid = colid + size(X,2);
% end
% DPhi = size(Phi,2);

% initialise top-level priors
an_prior         = opt.PriorParam{1};        % noise
bn_prior         = opt.PriorParam{2};   
if size(opt.PriorParam{4},1) > 1
    disp('using Wishart prior for the weights')
    n_prior = opt.PriorParam{3}; 
    P_prior = opt.PriorParam{4};
    Lmask   = tril(true(size(P_prior)));
    use_wishart = true;
else
    disp('using factorised ARD prior for weights');
    aa_prior = opt.PriorParam{3};        % precisions for the weights
    ba_prior = opt.PriorParam{4}; 
    use_wishart = false;
end

% initial posterior values
Theta = opt.X0_Theta;
beta  = Theta(1);
if use_wishart
    lambdavec = Theta(2:end);
else
    alpha = Theta(2:end);
end

% initialize posteriors
Theta_all = zeros(size(Theta,1), opt.nGibbsIter);
W_all     =  zeros(D,S, opt.nGibbsIter);

% initialize stats
stats.iter        = 1;
stats.opt         = opt;
stats.prior_theta = opt.PriorParam;
stats.arate_ell   = zeros(1,opt.nGibbsIter);
stats.arate_t2    = zeros(1,opt.nGibbsIter);

XX = X'*X;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Gibbs Sampling Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for g = 1:opt.nGibbsIter
    % display output
    if mod(g,update_interval) == 0
        disp(['Gibbs iter: ',num2str(g)]);
        stats.iter = g;
        
        if opt.PlotProgress, plot(Theta_all'); pause(0.1); end
    end
    
    % save output
    if mod(g,write_interval) == 0 && opt.WriteInterim && ...
            isfield(opt,'OutputFilename') && ...
            ~isempty(opt.OutputFilename)
        fprintf('Writing output ... ');
        save([opt.OutputFilename,'stats'],'stats');
        save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
        fprintf('done.\n');
    end
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample weights (W)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if use_wishart
        Ltril        = zeros(D);
        Ltril(Lmask) = lambdavec;
        Ldiag        = diag(Ltril);
        Lambda       = Ltril + Ltril' - diag(Ldiag);
    else
        Lambda = diag(alpha);
    end
    
    %Prec_w_post = beta*XX + Lambda;
    %C = beta*Prec_w_post\X';
    %cholC = chol(inv(Prec_w_post));
    
    Sigma_w_post = inv(beta*XX + Lambda);
    C = beta*Sigma_w_post*X';
    cholC = chol(Sigma_w_post);
    
    W = zeros(D,S);
    for s = 1:S
        %W(:,s) = C*Y(:,s);
        m = C*Y(:,s);
        W(:,s) = m + cholC * randn(D,1);
        W_all(:,s,g) = W(:,s);
    end
    %W_all(:,g) = W(:);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample noise precision (beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R  = Y-X*W;
    %bn = bn_prior + 0.5*trace(R'*R);
    bn = bn_prior + 0.5*sum(sum(R'.*R',2));
    an = an_prior + 0.5*S*N;
    
    % Draw from gamma
    beta = gamrnd(an,1/bn);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample weight precision
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if use_wishart 
        n = n_prior + S;
        P = P_prior + W*W';
        
        Lambda         = wishrnd(inv(P),n);
        Ld             = tril(Lambda);
        lambdavec      = Ld(Lmask);
        Theta_all(:,g) = [beta; lambdavec];
    else
        aa = aa_prior + 0.5*S;
        ba = ba_prior + 0.5*sum(W.*W,2);
        for d = 1:D
            alpha(d) = gamrnd(aa,1/ba(d));
        end
        
        % update Theta
        Theta_all(:,g)  = [beta; alpha];
    end
end
stats.time_taken       = toc;

if isfield(opt,'OutputFilename') && ~isempty(opt.OutputFilename)
    save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
    save([opt.OutputFilename,'W_all'],'W_all','-v7.3');
    save([opt.OutputFilename,'stats'],'stats','-v7.3');
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Private functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function x = trinvgamrnd(l,u,a,b)
% % sample from a truncated inverse gamma distribution
% 
% % check the range
% if l < 0; l = 0; end
% if u < 0; u = 1/eps; end
% 
% N = 10000;
% 
% ok = false;
% for n = 1:N
%     x = 1./gamrnd(a,1/b);
%     x = x(x > l & x < u);
%     if ~isempty(x)
%         ok = true;
%         break
%     end
% end
% if ~ok
%     error ('sampling from truncated Gamma failed');
% end
% end
