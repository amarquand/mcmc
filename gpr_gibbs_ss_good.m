function [stats] = gpr_gibbs(X, Y, opt)

% Sample a spatial random effects model where the model is specified as 
% a linear set of basis functions (currently an P degree polynomial
% expanision) plus a spatial Gaussian random field. The noise term is 
% currently an isotropic Gaussian.
%
% Written by A. Marquand
%
% Updates: 
% 20/10/15: Initial implementation
% 26/10/15: Revised implementation that: (i) uses a marginalised model for
%           the GRF and (ii) correctly models additive variance components.

% Specify functions needed for the matlab compiler
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
update_interval = 10;%0; 

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
an_prior         = opt.PriorParam{5};        % sn2 -> t2 (reparameterised)
bn_prior         = opt.PriorParam{6}; 
mu_beta_prior    = opt.PriorParam{7};        % beta (mean is assumed zero)
Prec_beta_prior  = (1./opt.PriorParam{8})*eye(DPhi); % for convenience 

% initial posterior values
Theta = opt.X0_Theta;
if any(regexp(func2str(opt.CovFunc{1}),'ard'))
    ell = Theta(1:D); parid = D;
else
    ell = Theta(1); parid = 1;
end
sf2   = Theta(parid+1);
sn2   = Theta(parid+2); 
%t2    = sn2 / sf2;
beta  = Theta(parid+3:end);  % theta is the first value sampled

% % compute initial covariance 
% if iscell(opt.CovFunc)
%     K  = feval(opt.CovFunc{:}, [log(ell); 0.5*log(sf2)], X)/sf2;
% else % old style
%     K  = feval(opt.CovFunc,X,Theta)/sf2;
% end
% % Cholesky decomposition (not including signal variance component)
% L_Ky = chol(K+sn2*eye(N))';

% initialize posteriors
Theta_all  = zeros(size(Theta,1),opt.nGibbsIter);

% initialize stats
stats.iter        = 1;
stats.opt         = opt;
stats.prior_theta = opt.PriorParam;
stats.arate_ell   = zeros(1,opt.nGibbsIter);
stats.arate_t2    = zeros(1,opt.nGibbsIter);

% log distributions
%logigam   = @(x,a,b) -(a+1).*x -b./x;
%logunif   = @(x,a,b) -log(x) - (x<a)/eps - (x>b)/eps;

% prior and likelihood specification
prior_sf2 = @(sf2,a,b) logigam(sf2,a,b);
prior_sn2 = @(sn2,a,b) logigam(sn2,a,b);
prior_ell = @(ell,a,b) logunif(ell,a,b);
lik_sf2   = @(sf2,ell,sn2,y,x,cov) likelihood_gaussian(y,x,[ell,sf2,sn2],cov);
lik_sn2   = @(sn2,ell,sf2,y,x,cov) likelihood_gaussian(y,x,[ell,sf2,sn2],cov);
lik_ell   = @(ell,sf2,sn2,y,x,cov) likelihood_gaussian(y,x,[ell,sf2,sn2],cov);

% posteriors (only used if sampling from the posterior directly)
post_sf2  = @(sf2,ell,sn2,y,x,cov,a,b) likelihood_gaussian(y,x,[ell,sf2,sn2],cov) + prior_sf2(sf2,a,b);
post_sn2  = @(sn2,ell,sf2,y,x,cov,a,b) likelihood_gaussian(y,x,[ell,sf2,sn2],cov) + prior_sn2(sn2,a,b);
post_ell  = @(ell,sf2,sn2,y,x,cov,a,b) likelihood_gaussian(y,x,[ell,sf2,sn2],cov) + prior_ell(ell,a,b);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Gibbs Sampling Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for g = 1:opt.nGibbsIter
    % display output
    if mod(g,update_interval) == 0
        disp(['Gibbs iter: ',num2str(g)]);
        stats.iter               = g;
        
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
        
   % slice sample the posterior directly
    %sf2 = slice_sample(sf2,post_sf2,[0 20], ell,sn2,y-Phi*beta,X,opt.CovFunc,af_prior,bf_prior);
    [sn2] = slice_sample(sn2,post_sn2,[0 20], ell,sf2,y-Phi*beta,X,opt.CovFunc,an_prior,bn_prior);
    ell = slice_sample(ell,post_ell,[al_prior,bl_prior],sf2,sn2,y-Phi*beta,X,opt.CovFunc,al_prior,bl_prior);   
      
    sf2 = slice_sample(sf2,lik_sf2,{@trinvgamrnd,{af_prior,bf_prior}},ell,sn2,y-Phi*beta,X,opt.CovFunc);
    %sn2 = slice_sample(sn2,lik_sn2,{@trinvgamrnd,{an_prior,bn_prior}},ell,sf2,y-Phi*beta,X,opt.CovFunc);
    %ell = slice_sample(sn2,lik_ell,{unifrnd, {al_prior,bl_prior}},sf2,sn2,y-Phi*beta,X,opt.CovFunc);
 
    K    = feval(opt.CovFunc{:}, [log(ell); log(sqrt(sf2))], X);
    L_Ky = chol(K+sn2*eye(N))';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample beta
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Prec_beta_post = Phi'*solve_chol(L_Ky',Phi) + Prec_beta_prior;
    mu_beta_post   = (Prec_beta_post\Phi')*solve_chol(L_Ky',y);
    
    % sample from posterior
    L_Sb = chol(inv(Prec_beta_post))';   % FIXME: not efficient
    beta = mu_beta_post + L_Sb*randn(DPhi,1);
      
    
    % update Theta    
    Theta           = [ell; sf2; sn2; beta];
    Theta_all(:,g)  = Theta;
end
stats.time_taken       = toc;

if isfield(opt,'OutputFilename') && ~isempty(opt.OutputFilename)
    save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
    save([opt.OutputFilename,'stats'],'stats','-v7.3');
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Private functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = logigam(x,a,b)
% log inverse gamma pdf
y = -(a+1).*x -b./x;
end 

function y = logunif(x,a,b)
% log uniform pdf 
y = -log(x) - (x<a)/eps - (x>b)/eps;
end



function x = trinvgamrnd(l,u,a,b,varargin)
% sample from a truncated inverse gamma distribution

if nargin > 4
    dim = [varargin{:}];
else
    dim = [1 1];
end

N = 100000;

%x = gamrnd(a,b,N,1);
x = 1./gamrnd(a,1/b,N,1);
x = x(x > l & x < u);
if isempty(x)
    error ('sampling from truncated Gamma failed');
end
x = x(1:prod(dim));
x = reshape(x,dim);
end

