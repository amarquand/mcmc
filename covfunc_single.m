function K = covfunc_single( K_all, LogTheta, knum)    

% covariance function that takes a simple sum of raw covariance kernels
% then outputs a weighted sum. old name: RMHMC_covFunc_SumK

Q = length(K_all);                   % number of data sources
N = size(K_all{1},1);                % number of samples
if mod(length(LogTheta), Q) == 0    
    n_gps = (length(LogTheta) / Q);      % number of GPs
else
    error('theta and K_all are incompatible sizes')
end


% Convert to a matrix (easier to deal with)
%LogTheta_m = reshape(LogTheta,Q,n_gps);

% Compute K
K = zeros(N*n_gps);
idx = 1:N; 
for c = 1:n_gps    
    K(idx,idx) = K_all{knum};   
    idx = idx + N;
end

%K = normalize_kernel(K);