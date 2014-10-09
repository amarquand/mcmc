function K = covfunc_scaled_dotprod(K_all, LogTheta)    

if iscell(K_all), K_all = K_all{:}; end

it2 = exp(-2*LogTheta);                                         % t2 inverse
K   = it2*K_all;

% derivatives:
%K = -2*it2*K;
