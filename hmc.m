function [E_accepted, X_accepted, R, accept_ratio, X_all] =  ...
    hmc(x, fx, opt, cholM, invM, varargin)

% Hamiltonian Monte Carlo method using the leapfrog algorithm based David
% MacKay's implementation notes (page 388). 
%
% usage: [E, x, r, accept_ratio] = HMC(X, fX, L, opt, cholM, invM, {function arguments})
%
% Inputs:
% X     : start coordinate for the simulation
% fX    : energy function (must return derivatives)
% opt   : options
%         opt.NumOfIterations    = length of the MCMC trajectory (L)
%         opt.NumOfLeapFrogSteps = number of leapfrog steps (Tau)
%         opt.StepSize           = mean step size (m_epsilon)
% cholM : Cholesky decomposition of the mass matrix
% invM  : Inverse of the mass matrix
%
% Outputs:
% E_accepted   : returned energy values
% X_accepted   : returned parameters
% R            : autocorrelation between successive samples
% accept_ratio : acceptance ratio
%
% Written by Andre Marquand
% Copyright (c) 2009 Andre Marquand

L         = opt.NumOfIterations;
Tau       = opt.NumOfLeapFrogSteps;
m_epsilon = opt.StepSize;

% get the initial function value and gradient 
[E, g] = feval(fx, x, varargin{:}); 

E_accepted = E;
E_all      = E;
X_accepted = x;
X_all      = x;
n_accept   = 0;

for l = 1:L
    % draw step size from an exponential distribution
    epsilon = -m_epsilon*log(1-rand);

    p = cholM*randn(size(x));                      % set initial momentum
    %p = randn(size(x));
    %p = (randn(1,size(x,1))*M)';
    H = p'*invM*p/2 + E;              % evaluate Hamiltonian H(x,p)
    
    % Leapfrog simulation
    %%%%%%%%%%%%%%%%%%%%%
    xnew = x; gnew = g;
    for tau = 1:Tau
        p = p - epsilon * gnew / 2;          % half-step in p
        xnew = xnew + epsilon * invM * p;       % half-step in x
        %xnew = xnew + epsilon * p;       % half-step in x
        
        [Enew, gnew] = ...
            feval(fx, xnew, varargin{:}); % get new gradient

        p = p - epsilon * gnew / 2;           % half-step in p
    end
    
    Enew = feval(fx, xnew, varargin{:});           % get new function value
    Hnew = p'*invM*p/2 + Enew;         % evaluate Hamiltonian H(x,p)
    dH = Hnew - H;                           % compute error in Hamiltonian

    if dH < 0
        accept = 1;
    elseif rand < exp(-dH)
        accept = 1;
    else
        accept = 0;
    end
    
    % Update new state
    %%%%%%%%%%%%%%%%%%
    if accept
        g = gnew;
        x = xnew;
        E = Enew;

        % book-keeping
        %%%%%%%%%%%%%%
        n_accept = n_accept+1;
        E_accepted = [E_accepted; E];
        X_accepted = [X_accepted x];
    end
    E_all = [E_all; Enew];
    X_all = [X_all x];

    % Stats
    %%%%%%%
    if length(E_accepted)>4 && ~mod(l,50)
        accept_ratio = n_accept / l;

        R = autocorrelation(E_all(isfinite(E_all)));
        %R = pacf(detrend(E_all),5,0);

        disp(['MCMC: l = ',num2str(l),'  acc_rate = ',num2str(accept_ratio,'%1.2f'), ...
            '  corr = ',num2str(R(1),'%1.2f')]);
    end
end
accept_ratio = n_accept / L;
R = autocorrelation(E_all(isfinite(E_all)));

return

% function to compute lag-1 autocorrelation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R = autocorrelation(X)

m = mean(X);
v = var(X);
n = length(X);
R = 0;
for t = 1:length(X) - 1
    R = R + (X(t)-m) * (X(t+1)-m);
end
R = R / ((n-1)*v);
return
