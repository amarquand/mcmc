function [stats] = gpr_gibbs(X, Y, opt)

%# function covLIN
%# function covLINard
%# function covSEiso
%# function covSEard
%# function covMaternard
%# function covMaterniso
%# function covSum
%# function covGrid
%# function infGrid
%# function sp_infGrid
%# function sp_covMTL

% seed random numbers
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

% housekeeping parameters
write_interval  = round(opt.nGibbsIter/10);
update_interval = 100; 

% make sure y is a vector
y     = Y(:); y(isnan(y)) = 0;

tic; % start timer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N,D] = size(X);  

% polynomial basis expansion for trend coefficients
Phi = zeros(size(X,1),size(X,2)*opt.DimPoly); colid = 1:size(X,2);
for d = 1:opt.DimPoly
    Phi(:,colid) = X.^d; colid = colid + size(X,2);
end
DPhi = size(Phi,2);

% initialise top-level priors
al_prior         = opt.PriorParam{1};        % ell
bl_prior         = opt.PriorParam{2};     
af_prior         = opt.PriorParam{3};        % sf2
bf_prior         = opt.PriorParam{4}; 
an_prior         = opt.PriorParam{5};        % sn2
bn_prior         = opt.PriorParam{6}; 
mu_beta_prior    = opt.PriorParam{7};        % beta
S_beta_prior     = opt.PriorParam{8}*eye(DPhi);
Prec_beta_prior  = (1./opt.PriorParam{8})*eye(DPhi); % for convenience 

% initial posterior values
Theta = opt.X0_Theta;
f     = opt.X0_f;
if any(regexp(func2str(opt.CovFunc{1}),'ard'))
    ell = Theta(1:D); parid = D;
else
    ell = Theta(1); parid = 1;
end
sf2   = Theta(parid+1);
%sn2   = Theta(parid+2); % sn2 is the first variable sampled
beta  = Theta(parid+3:end); 

% compute initial covariance 
if iscell(opt.CovFunc)
    K  = feval(opt.CovFunc{:}, [log(ell); log(0.5*sf2)], X);
else % old style
    K  = feval(opt.CovFunc,X,Theta);
end

% initial posterior for S
Prec_beta_post  = Phi'*Phi + Prec_beta_prior;

% compute initial posterior for beta by setting to prior mean
mu_beta_post = mu_beta_prior*ones(DPhi,1); % = zeros(DimPoly,1)

% initialize posteriors
f_all      = zeros(size(f,1),opt.nGibbsIter);   fidx = 1;
alpha_all  = zeros(size(f,1),opt.nGibbsIter); % saves time for regression tasks
Theta_all  = zeros(size(Theta,1),opt.nGibbsIter);

% initialize stats
stats.iter        = 1;
stats.opt         = opt;
stats.prior_theta = opt.PriorParam;
stats.arate_cov   = zeros(1,opt.nGibbsIter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Gibbs Sampling Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc_cov_all = 0; gidx = 1:update_interval; 
for g = 1:opt.nGibbsIter
    %g
    % display output
    if mod(g,update_interval) == 0
        arate_cov     = acc_cov_all / update_interval;
        
        disp(['Gibbs iter: ',num2str(g),' arate(cov)=',num2str(arate_cov,'%2.2f')]);
        acc_cov_all = 0;
        
        % update stats
        stats.iter               = g;
        stats.arate_cov(gidx)    = arate_cov;   
        gidx = gidx + update_interval;
        
        if opt.PlotProgress, plot(Theta_all'); pause(0.1); end
    end 
   
    % save output
    if mod(g,write_interval) == 0 && opt.WriteInterim && ...
       isfield(opt,'OutputFilename') && ...
       ~isempty(opt.OutputFilename)
        fprintf('Writing output ... ');
        save([opt.OutputFilename,'stats'],'stats');        
        save([opt.OutputFilename,'f_all'],'f_all','-v7.3');
        save([opt.OutputFilename,'alpha_all'],'alpha_all','-v7.3');
        save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
        fprintf('done.\n');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample noise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    an = an_prior + 0.5*N;
    bn = bn_prior + 0.5*((y-f)'*(y-f) + mu_beta_post'/Prec_beta_post*mu_beta_post);
    % Note: set f=0 in the line above to get BLR    
 
    % Draw new sn2 from inverse gamma
    sn2   = 1./gamrnd(an,1/bn);
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample f
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % compute & solve cholesky (alpha = inv(C)*y);
    L_Ky  = chol(K+sn2*eye(N))';
    alpha = solve_chol(L_Ky',y-Phi*beta);
    
    mu_f_post = K*alpha;
    v         = L_Ky\K;
    S_f_post  = K - v'*v;
        
    % sample from posterior
    L_Sf  = chol(S_f_post)';
    f     = mu_f_post + L_Sf*randn(N,1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample beta
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          
    % compute posterior using the matrix inversion lemma
    Prec_beta_post = Phi'*Phi + Prec_beta_prior;
    mu_beta_post   = Prec_beta_post\Phi'*(y-f);
    
    % sample from posterior    
    L_Sb = chol(inv(Prec_beta_post)*sn2)';                  % FIXME
    beta = mu_beta_post + L_Sb*randn(DPhi,1);
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample covariance scale
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    af = af_prior + 0.5*N;
    C  = (K+sn2*eye(N)+sn2*Phi*S_beta_prior*Phi')/sf2;
    bf = bf_prior + 0.5*f'/C*f;
    
    % Draw new sf2 from inverse gamma
    sf2   = 1./gamrnd(af,1/bf);
    
    if iscell(opt.CovFunc)
        K  = feval(opt.CovFunc{:}, [log(ell); log(0.5*sf2)], X);
    else % old style
        K  = feval(opt.CovFunc,X,Theta);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample lengthscale parameter(s)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %nu = (L_Ky\f);
    %f_new     = L_Kt_new*nu;
    logell = log(ell);
    
    % make a step in theta
    logell_new = logell + opt.mh.StepSize*(eye*randn(length(logell),1));
    
    % compute new covariance
    if iscell(opt.CovFunc)
        K_new  = feval(opt.CovFunc{:}, [logell_new; log(0.5*sf2)], X);
    else % old style
        K_new  = feval(opt.CovFunc, X, [logell_new; log(0.5*sf2)]);
    end
   
    % compute (old) log marginal likelihood
    Csf2   = K+sn2*eye(N)+sn2*Phi*S_beta_prior*Phi';
    L_Csf2 = chol(Csf2)';
    LogLik = y'*solve_chol(L_Csf2',y) - sum(log(diag(L_Csf2))) - 0.5*log(2*pi);
     
    % compute (new) log marginal likelihood
    Csf2_new   = K_new+sn2*eye(N)+sn2*Phi*S_beta_prior*Phi';
    L_Csf2_new = chol(Csf2_new)';
    LogLik_new = y'*solve_chol(L_Csf2_new',y) - sum(log(diag(L_Csf2_new))) - 0.5*log(2*pi);
    
    % compute priors for new and old paramters
    LP_cov     = zeros(size(logell,1),1);
    LP_cov_new = zeros(size(logell,1),1);
    const      = al_prior*log(bl_prior) - gammaln(al_prior);
    % lengthscale parameters
    for i = 1:length(LP_cov)
        LP_cov(i)     = const + (al_prior-1)*logell(i) - bl_prior*exp(logell(i));
        LP_cov_new(i) = const + (al_prior-1)*logell_new(i) - bl_prior*exp(logell_new(i));
    end
    % Note: we do not need to include p(sf2|a,b) because they are constant
    LP_cov     = sum(LP_cov);
    LP_cov_new = sum(LP_cov_new);
        
    Ratio = (LogLik_new + LP_cov_new) - (LogLik + LP_cov);
    
    if Ratio > 0 || (Ratio > log(rand)) % accept
        % update theta
        fprintf('iteration %d: accept (acc rate = %2.2f)\n',g,(acc_cov_all+1)/(mod(g,update_interval)+1))
        acc_cov = 1;
        ell     = exp(logell_new);
        K       = K_new;   
        % recompute alpha
        L_Ky    = chol(K+sn2*eye(N))';
        alpha   = solve_chol(L_Ky',y-Phi*beta);    

    else % reject
        fprintf('iteration %d: reject (acc rate = %2.2f)\n',g,acc_cov_all/(mod(g,update_interval)+1))
        acc_cov = 0;
    end
    acc_cov_all = acc_cov_all + acc_cov;
    
    Theta = [ell; sf2; sn2; beta];
        
    % save posteriors and kernel weights
    Theta_all(:,g)    = Theta;
    f_all(:,fidx)     = f;
    alpha_all(:,fidx) = alpha;
    fidx              = fidx + 1;
end
stats.time_taken       = toc;
stats.arate_cov_mean   = mean(stats.arate_cov);

disp(['Mean acceptance rate (cov): ',num2str(stats.arate_cov_mean,'%2.2f')]);

if isfield(opt,'OutputFilename') && ~isempty(opt.OutputFilename)
    save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
    save([opt.OutputFilename,'f_all'],'f_all','-v7.3');
    save([opt.OutputFilename,'alpha_all'],'alpha_all','-v7.3');
    save([opt.OutputFilename,'stats'],'stats','-v7.3');
end
end

