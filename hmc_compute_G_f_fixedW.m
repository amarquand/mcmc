function [G InvG CholG GDeriv] = RMHMC_compute_G_f_fixedW(f,InvK, T)

N     = size(T,1);       % number of samples
C     = size(T,2);       % number of classes
n_gps = length(f)/N;     % number of GPs
D     = N*n_gps;         % dimensionality

% Pre-allocate memory for partial derivatives
persistent GD initialized
if ~initialized
    if nargout > 3
        GD = cell(1,D);
        for d = 1:D
            GD{d} = zeros(D,D);
        end
        initialized = 1;
    end
end

%%%%%%%%%%%
% Compute G
%%%%%%%%%%%
%P = RMHMC_likelihood_f(f,T);
P = (1/C)*ones(N,n_gps);
p = reshape(P,N*n_gps,1);

invA = zeros(D);
row_idx = zeros(D,1);
col_idx = repmat((1:N)',n_gps,1);
for c = 1:n_gps
    idx = (c-1)*N+1:c*N;
    
    invA(idx,idx) = inv( InvK(idx,idx) + diag(p(idx)) );
   
    row_idx(idx) = idx;   
end
Pi = sparse(row_idx,col_idx,p);

G = InvK + diag(p) - Pi*Pi';

if nargout > 1                                   % do we need to invert G?
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Invert G using Matrix Inversion Lemma
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %InvG = inv(G + 1e-5*eye(size(G,1)));
    InvG = invA - invA*Pi*inv(-eye(size(Pi,2)) + Pi'*invA*Pi)*Pi'*invA;
    
    if nargout > 2                % do we need the Cholesky decomposition?
        %%%%%%%%%%%%%%%%%%
        % Compute Cholesky
        %%%%%%%%%%%%%%%%%%
        
        CholG = chol(G)';
        %CholG = chol(G + 1e-5*eye(size(G,1)))';
        
        if nargout > 3
            %%%%%%%%%%%%%%%%%%%%%
            % Compute Derivatives
            %%%%%%%%%%%%%%%%%%%%%
            for d = 1:D
                class = ceil(d/N); %which class are we working on?
                if rem(d,N) == 0, sample = N; else sample = rem(d,N); end
                
                % compute dP
                dP = zeros(N,n_gps);
                for c = 1:n_gps
                    if c == class
                        dP(sample,c) = P(sample,class).*(1-P(sample,class));
                    else
                        dP(sample,c) = -P(sample,class).*P(sample,c);
                    end
                end
                dp = reshape(dP,N*n_gps,1);
                dpnz = dp(abs(dp) > 0);
                
%                 % now compute V and Pi (method 1)
%                 Pi = zeros(D,N);
%                 V  = zeros(D,N);
%                 for c = 1:n_gps
%                    idx = (c-1)*N+1:c*N;
%                 
%                    Pi(idx,:) = diag(P(:,c));
%                    V(idx,:) = diag(dP(:,c));
%                 end
%                 gd1 = diag(dp) - Pi*V' - V*Pi'; % slow
                
                % method 3 (fastest)
                piv_row_idx = zeros(n_gps^2,1);
                piv_col_idx = zeros(n_gps^2,1);
                p_idx = zeros(n_gps,1);
                piv = zeros(n_gps^2,1);
                ccount = 1;
                for c = 1:n_gps
                    for cc = 1:n_gps
                        piv_row_idx(ccount) = (c-1)*N + sample;
                        piv_col_idx(ccount) = (cc-1)*N + sample;
                        
                        piv(ccount) = P(sample,c)*dP(sample,cc);
                        
                        ccount = ccount + 1;
                    end
                    p_idx(c) = (c-1)*N + sample;
                end
                % Do the speed test on this again...
                PiV = sparse(piv_row_idx, piv_col_idx, piv, D, D);
                gd = sparse(p_idx,p_idx,dpnz,D,D) - PiV - PiV';
                
                GD{d} = gd;          % separated for speed diagnostics
                
                % debug
                %if max(max(gd1 - gd)) ~=0
                %    disp (['differential ',num2str(max(max(gd1 - gd)))])
                %end
            end
            
            % set output value
            GDeriv = GD;
        end
    end
end

end

