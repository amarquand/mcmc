function [P, logLik, dlogLik] = likelihood_multinomial(f,T) 

% old name: RMHMC_likelihood_f

N     = size(T,1);
C     = size(T,2);
n_gps = length(f)/N;  

if n_gps == C-1
    eF = exp(reshape(f,N,C-1));
    Z = sum(eF,2)+1;
elseif n_gps == C
    eF = exp(reshape(f,N,C));
    Z = sum(eF,2);
else
    error('length of latent function does not agree with size of labels');
end

P = eF ./ repmat(Z,1,size(eF,2));
p = reshape(P,N*n_gps,1);           % N.B.: may not include reference class               
    
t = reshape(T(:,1:n_gps),N*n_gps,1);

logLik  = t'*f - sum(log(Z));

if nargout > 2
    dlogLik = t - p;
end

end