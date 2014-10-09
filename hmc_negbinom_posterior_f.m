function [LogPost, dLogPost] = hmc_negbinom_posterior_f(f, T, InvC, LogDetC,LogBinomVar)

if nargout == 1
    [tmp, LogLik] = likelihood_negbinomial(f, T,LogBinomVar);
    LogPrior      = prior_log_gauss(f,InvC,LogDetC);
else
    [tmp, LogLik, dLogLik] = likelihood_negbinomial(f, T,LogBinomVar);
    [LogPrior, dLogPrior]  = prior_log_gauss(f,InvC,LogDetC);
    
    dLogPost = -(dLogLik + dLogPrior);
end

LogPost = -(LogLik + LogPrior);
        
        