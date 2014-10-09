function K = covfunc_wsum(K_all, LogTheta, B)    

% covariance function that takes a simple sum of raw covariance kernels
% then outputs a weighted sum. old name: RMHMC_covFunc_SumK

Q = length(K_all);                   % number of data sources
N = size(K_all{1},1);                % number of samples
if mod(length(LogTheta), Q) == 0    
    n_gps = (length(LogTheta) / Q);      % number of GPs
else
    error('theta and K_all are incompatible sizes')
end

% Accommodate the case where weighting factors are the same for all classes
if length(LogTheta) == Q
   LogTheta = repmat(LogTheta,Q*n_gps,1); 
end
    
% Convert to a matrix (easier to deal with)
LogTheta_m = reshape(LogTheta,Q,n_gps);

% Compute K
K = zeros(N*n_gps);
idx = 1:N; 
for c = 1:n_gps
    K_sum = zeros(size(K_all,1));
    for q = 1:Q
        K_sum = K_sum + exp(LogTheta_m(q,c)) * K_all{q};
    end
    
    K(idx,idx) = K_sum;   
    idx = idx + N;
end

if nargin > 3 % are there test data?
    if strcmp(B,'diag') && numel(B)>0;   % determine mode
        K = diag(K);
    else
        error ('covfunc_wsum: unknown mode');
    end
end