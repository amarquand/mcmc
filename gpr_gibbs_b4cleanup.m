function [stats] = gp_gpr_gibbs(X, Y, opt)

% seed random numbers
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

% housekeeping parameters
write_interval  = 50;%round(opt.nGibbsIter/10);
update_interval = 200; 

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
acc_cov_all = 0; gidx = 1:50; lpcov  = []; lplik = [];
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
        
        plot(Theta_all'); pause(0.1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample noise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %bn = bn_prior + 0.5*sum((y - f - Phi*beta).^2)
    %bn = bn_prior + 0.5*(y'*y -2*(Phi*beta+f)'*y + (Phi*beta+f)'*(Phi*beta+f));
    % from bishop:
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
    
    %muf_post = muf + Sr*alpha;
    mu_f_post = K*alpha; % + muf
    v         = L_Ky\K;
    S_f_post  = K - v'*v;
        
    L_Sf  = chol(S_f_post)';
    f     = mu_f_post + L_Sf*randn(N,1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample beta
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          
    % compute posterior using the matrix inversion lemma
    %%S_beta_post  = inv(Phi'*Phi/sn2 + Prec_beta_prior);
    %%mu_beta_post = (Phi'*Phi/sn2 + Prec_beta_prior)\Phi'*(y-f); %S*InvKy*y;
    
    Prec_beta_post = Phi'*Phi + Prec_beta_prior;
    mu_beta_post   = Prec_beta_post\Phi'*(y-f); %S*InvKy*y;
    %S_beta_post  = inv((Phi'*Phi + Prec_beta_prior)/sn2 + 1e-5*eye(DPhi));
    %mu_beta_post = ((Phi'*Phi + Prec_beta_prior)/sn2+ 1e-5*eye(DPhi))\Phi'*(y-f); %S*InvKy*y;
    
    % sample from posterior    
    L_Sb = chol(inv(Prec_beta_post)*sn2)';
    beta = mu_beta_post + L_Sb*randn(DPhi,1);
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample covariance scale
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    af = af_prior + 0.5*N;
    %bf = bf_prior + 0.5*f'/K*f;
    %%bf = bf_prior + 0.5*f'*solve_chol(K',f);
    C = (K+sn2*eye(N)+sn2*Phi*S_beta_prior*Phi')/sf2;
    bf = bf_prior + 0.5*f'/C*f;
    
    % Draw new sn2 from inverse gamma & update parameter vector
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
    
    % compute covariance
    if iscell(opt.CovFunc)
        K_new  = feval(opt.CovFunc{:}, [logell_new; log(0.5*sf2)], X);
    else % old style
        K_new  = feval(opt.CovFunc, X, [logell_new; log(0.5*sf2)]);
    end
    %L_Ky_new  = chol(K_new+sn2*eye(N))';
    %alpha_new = solve_chol(L_Ky_new',y-Phi*beta);
    
    % compute (old) log marginal likelihood using R/W eq 2.44 (p 29)
    % iKyX = solve_chol(L_Ky',Phi);
    % A    = Prec_beta_prior + Phi'*iKyX;
    % C    = iKyX/A*iKyX';
    % LogLik   = -0.5*y'*alpha + 0.5*y'*C*y - sum(log(diag(L_Ky))) ...
    %           -0.5*log(det(S_beta_prior)) -0.5*log(det(A)) -0.5*N*log(2*pi);
    Csf2   = K+sn2*eye(N)+sn2*Phi*S_beta_prior*Phi';
    L_Csf2 = chol(Csf2)';
    LogLik = y'*solve_chol(L_Csf2',y) - sum(log(diag(L_Csf2))) - 0.5*log(2*pi);
     
    % compute (new) log marginal likelihood
    % iKyX_new   = solve_chol(L_Ky_new',Phi);
    % A_new      = Prec_beta_prior + Phi'*iKyX_new;
    % C_new      = iKyX_new/A_new*iKyX_new';
    % LogLik_new = -0.5*y'*alpha_new + 0.5*y'*C_new*y - sum(log(diag(L_Ky_new))) ...
    %     -0.5*log(det(S_beta_prior)) -0.5*log(det(A_new)) -0.5*N*log(2*pi);
    Csf2_new   = K_new+sn2*eye(N)+sn2*Phi*S_beta_prior*Phi';
    L_Csf2_new = chol(Csf2_new)';
    LogLik_new = y'*solve_chol(L_Csf2_new',y) - sum(log(diag(L_Csf2_new))) - 0.5*log(2*pi);
    
    % compute priors for new and old paramters
    LP_cov     = zeros(size(logell,1)+2,1);
    LP_cov_new = zeros(size(logell,1)+2,1);
    const      = al_prior*log(bl_prior) - gammaln(al_prior);
    % lengthscale parameters
    for i = 1:length(LP_cov)-2
        LP_cov(i)     = const + (al_prior-1)*logell(i) - bl_prior*exp(logell(i));
        LP_cov_new(i) = const + (al_prior-1)*logell_new(i) - bl_prior*exp(logell_new(i));
    end
    % signal variance (gamma)
    % const = af_prior*log(bf_prior) - gammaln(af_prior);
    % LP_cov(i+1)     = const + (af_prior-1)*lh(i+1) - bf_prior*exp(lh(i+1));
    % LP_cov_new(i+1) = const + (af_prior-1)*lh(i+1) - bf_prior*exp(lh(i+1));
    
    % signal variance (inverse gamma)
    const = af_prior*log(bf_prior) - gammaln(af_prior);
    LP_cov(i+1)     = const - (af_prior+1)*log(sf2) - bf_prior/sf2;
    LP_cov_new(i+1) = const - (af_prior+1)*log(sf2) - bf_prior/sf2;
           
    LP_cov          = sum(LP_cov);
    LP_cov_new      = sum(LP_cov_new);
    
    %lpcov = [lpcov LP_cov];
    %lpcov = [lplik LogLik];
    
    Ratio = (LogLik_new + LP_cov_new) - (LogLik + LP_cov);
    
    if Ratio > 0 || (Ratio > log(rand)) % accept
        % update theta
        fprintf('iteration %d: accept (acc rate = %2.2f)\n',g,(acc_cov_all+1)/mod(g,update_interval))
        ell     = exp(logell_new);
        %sf2     = sf2_new; %exp(2*lh_new(end));
        %L_Ky    = L_Ky_new; % not needed as it is recomputed anyway
        K       = K_new;
        %alpha   = alpha_new;
        acc_cov = 1;
    else % reject
        fprintf('iteration %d: reject (acc rate = %2.2f)\n',g,acc_cov_all/mod(g,update_interval))
        acc_cov = 0;
    end
    acc_cov_all = acc_cov_all + acc_cov;
    
    Theta = [ell; sf2; sn2; beta];
        
    % save posteriors and kernel weights for regression
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

