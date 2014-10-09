function [LogPost, dLogPost] = hmc_posterior_f(f, T, InvC, LogDetC, mu)

if nargin < 5
    mu = 0;
end

if nargout == 1
    [tmp, LogLik] = likelihood_multinomial(f, T);
    LogPrior      = prior_log_gauss(f,InvC,LogDetC,mu);
else
    [tmp, LogLik, dLogLik] = likelihood_multinomial(f, T);
    [LogPrior, dLogPrior]  = prior_log_gauss(f,InvC,LogDetC,mu);
    
    dLogPost = -(dLogLik + dLogPrior);
end

LogPost = -(LogLik + LogPrior);
        
        