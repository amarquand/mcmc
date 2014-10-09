function [P Pt Phn]  = gpc_pred_gibbs_rmhmc_mh(X, tr, te, Y, opt)

% Subject and cross-validation parameters
[n,k] = size(Y);
%n_gps = k;
ntr   = length(tr);
nte   = length(te);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin k-fold cross-validation block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the posteriors
disp('++ Loading posteriors ...');
load([opt.OutputFilename,'f_all'])
load([opt.OutputFilename,'LogTheta_all'])

f_post        = f_all(:,opt.BurnIn:opt.TestInterval:end);
LogTheta_post = LogTheta_all(:,opt.BurnIn:opt.TestInterval:end);

n_gps = size(f_post,1)/ntr;

clear f_all LogTheta_all
    
% Configure training & test kernel matrices
ktr= []; kte = [];
for c = 1:n_gps;
    ktr = [ktr tr+(c-1)*n];
    kte = [kte te+(c-1)*n];
end

% Compute predictions
disp('++ Computing predictions ...');
P  = zeros(nte, k);
Pt = zeros(ntr, k);
n_test_samples = 0;
for i = 1 : size(f_post,2);
    %Kt  = feval(opt.CovFunc, X, LogTheta_post(:,i));
    %Ks  = feval(opt.CovFunc, Xs, LogTheta_post(:,i));
    %Kss = feval(opt.CovFunc, Xss, LogTheta_post(:,i), 'diag');
    K   = feval(opt.CovFunc, X, LogTheta_post(:,i));
    Kt  = K(ktr,ktr);
    Ks  = K(kte,ktr);
    Kss = K(kte,kte);
    InvKt = invert_K(Kt,Y(tr,:));
    
    if opt.nTestSamples > 1
        P_i = zeros(nte, k);
        for s = 1:nte
            
            k_idx = 1:n;
            q_idx = s;
            Qs  = zeros(n_gps,n*n_gps);
            Qss = zeros(n_gps);
            for q = 1:n_gps
                Qs(q,k_idx) = Ks(q_idx,k_idx);
                Qss(q,q) = Kss(q_idx,q_idx);
                
                q_idx = q_idx + nte;
                k_idx = k_idx + n;
            end
            
            mus = Qs*InvKt*f_post(:,i);
            Ss  = Qss - Qs*InvKt*Qs';
            Ls  = chol( Ss )';%+ 1e-5*eye(size(Ss)) )';
            
            P_is = zeros(1,k);
            for j = 1:opt.nTestSamples
                fs = mus + Ls*randn(size(mus));
                %P = likelihood_multinomial(fs,Y(test(s),:));
                efs = exp(fs)';
                if n_gps == k -1
                    Z = sum(efs,2)+1;
                    Ps = efs ./ Z;
                    Ps = [Ps 1-sum(Ps,2)];
                else
                    Z = sum(efs,2);
                    Ps = efs ./ Z;
                end
                
                P_is = P_is + Ps;
            end
            P_i(s,:) = P_is / opt.nTestSamples;
        end
    else
        % compute predictive mean and variance
        mu = Ks*InvKt*f_post(:,i);
        %Sigma = Kss - Ks*InvKt*Ks';
        
        P_i = likelihood_multinomial(mu,Y(te,:));
        
        % predictions on training set
        Pt_i = likelihood_multinomial(Kt*InvKt*f_post(:,i),Y(tr,:));
        if n_gps < k
            P_i = [P_i 1-sum(P_i,2)];
            Pt_i = [Pt_i 1-sum(Pt_i,2)];
        end
    end
    P = P + P_i;
    Pt= Pt + Pt_i;
    
    n_test_samples = n_test_samples  + 1;
end
P  = P ./ n_test_samples;
Pt = Pt./ n_test_samples;

% renormalize
phn = zeros(nte,k);
for cl = 1:k
    [ptrs,iptrs]=sort(Pt(:,cl));
    [ptes,iptes]=sort(P(:,cl));
    %P(:,ips)=pts;
    h = spline(1:ntr,ptrs,linspace(1,ntr,nte));
    phn(iptes,cl) = h';
end
Phn = phn ./ repmat(sum(phn,2),1,k);


