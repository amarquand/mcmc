function [LogPost, dLogPost] = rmhmc_posterior_f(f, T, InvC, LogDetC)

if nargout == 1
    [tmp, LogLik] = likelihood_multinomial(f, T);
    LogPrior      = prior_log_gauss(f,InvC,LogDetC);
else
    [tmp, LogLik, dLogLik] = likelihood_multinomial(f, T);
    [LogPrior, dLogPrior]  = prior_log_gauss(f,InvC,LogDetC);
    
    dLogPost = dLogLik + dLogPrior;
end

LogPost = LogLik + LogPrior;
        
        