function [p, logLik, dlogLik] = likelihood_poisson(f,y,scale) 

% computes a Poisson likelihood given by:
%
% p(y|f) = (exp(-f)*f^y) / y!
%

n = length(y);

%mu = scale * exp(f);
mu = exp(f);

Z  = gamma(y+1);  % = factorial(y+1)
p  = exp(-mu).*mu.^y ./ Z;

%logLik  = sum(real(y.*log(mu)) - mu - Z);
logLik = y'*f - sum(mu) -sum(Z);

if nargout > 2
    %dlogLik = y ./ f - 1;
    dlogLik = y - mu;
end

end