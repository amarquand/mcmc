function [p, logLik, dlogLik] = likelihood_negbinomial(f,y,logr) 

% computes a Poisson likelihood given by:
%
% p(y|f) = (exp(-f)*f^y) / y!
%

r = exp(logr);
%n = length(y);

mu = exp(f);

Z  = gamma(r+y) ./ (gamma(y+1).* gamma(r));  % = factorial(y+1)
p  = Z .* (r./(r+mu)).^r.*(mu./(r+mu)).^y;

logLik = sum ( gammaln(r+y) - gammaln(y+1) -gammaln(r) + ...
         r.*(log(r) - log(r+mu)) + y.*(log(mu) - log(r+mu)));

if nargout > 2
    dlogLik = -mu.*(r+y) ./ (r+mu) + y;%sum(y - mu);
end

end