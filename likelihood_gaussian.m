function [LL, L] = likelihood_gaussian(y,X,theta,cov)

persistent last_hyp 
persistent last_L
persistent last_LL
persistent last_y

N      = size(y,1);
logell = log(theta(1));
logsf  = log(sqrt(theta(2)));
sn2    = theta(3);
hyp    = [logell;logsf;sn2];

if isempty(last_hyp); last_hyp = zeros(size(hyp)); end
if isempty(last_y);   last_y   = nan(N,1); end

% compute Cholesky if hyperparameters have changed
if nargin < 5 || any(last_hyp ~= hyp)
    K  = feval(cov{:}, [logell; logsf], X);
    L  = chol(K+sn2*eye(N))';
else
    L = last_L;
end

if isempty(last_LL) || any(last_L(:) ~= L(:)) || any(y ~= last_y)
    LL = -0.5*y'*solve_chol(L',y) -sum(log(diag(L))) - 0.5*N*log(2*pi);
else
    LL = last_LL;
end

last_L   = L;
last_LL  = LL;
last_hyp = hyp;
last_y   = y;

% if nargin < 5 || any(last_hyp ~= hyp)
%     K  = feval(cov{:}, [logell; logsf], X);
%     L  = chol(K+sn2*eye(N))';
%     last_hyp = hyp;
%     last_L   = L;
%     %if y ~= last_y
%         LL = -0.5*y'*solve_chol(L',y) -sum(log(diag(L))) - 0.5*N*log(2*pi);
%         last_LL  = LL;
%         last_y   = y;
%     %else
%     %    LL = last_LL;
%     %end
% else
%     L = last_L;
%     LL = last_LL;
% end

% else
%     L  = last_L;
%     LL = last_LL;
% end
%fprintf('hyp = %2.6f %2.6f %2.6f \n',hyp(1),hyp(2),hyp(3))
end
