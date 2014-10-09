function [LogPost, dLogPost] = hmc_poiss_posterior_f(f, T, InvC, LogDetC,PoissScale)

if nargout == 1
    [tmp, LogLik] = likelihood_poisson(f, T,PoissScale);
    LogPrior      = prior_log_gauss(f,InvC,LogDetC);
else
    [tmp, LogLik, dLogLik] = likelihood_poisson(f, T,PoissScale);
    [LogPrior, dLogPrior]  = prior_log_gauss(f,InvC,LogDetC);
    
    dLogPost = -(dLogLik + dLogPrior);
end

LogPost = -(LogLik + LogPrior);
        
        