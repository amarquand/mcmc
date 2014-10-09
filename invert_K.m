function [InvK, LogDetK, L_K] = RMHMC_invertK(K,T)

N     = size(T,1);
%C     = size(T,2);
n_gps = size(K,1)/N; 

InvK = zeros(N*n_gps);
%ridge = 1e-5*eye(N);
ridge = 1e-3*eye(N);
dets = zeros(n_gps,1);
for c = 1:n_gps
    idx = (c-1)*N+1:c*N;
    
    InvK(idx,idx) = inv(K(idx,idx)+ridge);
    dets(c) = prod( diag(chol(K(idx,idx)+ridge)).^2 );
    
end
LogDetK = log(prod(dets));
    
% Kr = K+1e-5*eye(size(K,1));
% InvK = inv(Kr);
% LogDetK = log(det(Kr));

if nargout > 2
    ridge_chol = 1e-5*eye(size(K,1));
    L_K = chol(K+ridge_chol)';
end