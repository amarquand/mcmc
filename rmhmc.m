function [XSaved, XAccepted] = rmhmc(X, fx, fxargs, gx, gxargs, opt)
    
D = length(X);

% Begin RMHMC block
%%%%%%%%%%%%%%%%%%%

Proposed = 0;
Accepted = 0;
Aidx = 1;

% set initial value for X
XSaved = zeros(opt.NumOfIterations,D);
XAccepted = zeros(opt.NumOfIterations,D);
warning off

% Pre-allocate memory for partial derivatives
for d = 1:D
    InvGdG{d} = zeros(D);
end

% Calculate f(X) for current X
CurrentfX = feval(fx, X, fxargs{:});

for IterationNum = 1:opt.NumOfIterations
    
    if mod(IterationNum,50) == 0
        disp([num2str(IterationNum) ' iterations completed, acceptance rate: ',num2str(Accepted/Proposed)])
        Proposed = 0;
        Accepted = 0;
        drawnow
    end
    
    %IterationNum
    
    XNew = X;
    Proposed = Proposed + 1;
    
    % Calculate G 
    [G InvG CholG GDeriv] = feval(gx, XNew, gxargs{:});
    % and the partial derivatives dG/dX
    for d = 1:D
        InvGdG{d}      = InvG*GDeriv{d};
        TraceInvGdG(d) = trace(InvGdG{d});
    end
    
    OriginalG     = G;
    OriginalCholG = CholG;
    OriginalInvG  = InvG;
    
    %ProposedMomentum = (randn(1,D)*OriginalCholG)';
    ProposedMomentum = OriginalCholG*randn(D,1);
    OriginalMomentum = ProposedMomentum;
   
    if (randn > 0.5) TimeStep = 1; else TimeStep = -1; end
    
    RandomSteps = ceil(rand*opt.NumOfLeapFrogSteps);
    
    SavedSteps(1,:) = XNew;
        
    % Perform leapfrog steps
    for StepNum = 1:RandomSteps
        
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        [tmp, dLdTheta] = feval(fx, XNew, fxargs{:});
        TraceTerm = 0.5*TraceInvGdG';
        
        % Multiple fixed point iteration
        PM = ProposedMomentum;
        for FixedIter = 1:opt.NumOfNewtonSteps
            MomentumHist(FixedIter,:) = PM;
            
            InvGMomentum = InvG*PM;
            for d = 1:D
                LastTerm(d)  = 0.5*(PM'*InvGdG{d}*InvGMomentum);
            end
            
            PM = ProposedMomentum + TimeStep*(opt.StepSize/2)*(dLdTheta - TraceTerm + LastTerm');
        end
        ProposedMomentum = PM;
           
        %%%%%%%%%%%%%%%%%%%%%%%
        % Update f parameters %
        %%%%%%%%%%%%%%%%%%%%%%%
        %%% Multiple Fixed Point Iterations %%%
        OriginalInvGMomentum  = G\ProposedMomentum;
        
        PX = XNew;
        for FixedIter = 1:opt.NumOfNewtonSteps
            XHist(FixedIter,:) = PX;
            
            G = feval(gx, PX, gxargs{:});
            
            InvGMomentum = G\ProposedMomentum;
            
            PX = XNew + (TimeStep*(opt.StepSize/2))*OriginalInvGMomentum + (TimeStep*(opt.StepSize/2))*InvGMomentum;
        end
        XNew = PX;

        % Calculate G  based on new parameters
        [G InvG CholG GDeriv] = feval(gx, XNew, gxargs{:});
        % Calculate dG/dX
        for d = 1:D
            InvGdG{d}      = InvG*GDeriv{d};
            TraceInvGdG(d) = trace(InvGdG{d});
        end
        
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        % Calculate last term in dH/dTheta
        InvGMomentum = (InvG*ProposedMomentum);
        for d = 1:D
            LastTerm(d) = 0.5*((ProposedMomentum'*InvGdG{d}*InvGMomentum));
        end
        
        % Calculate dH/dTheta
        [tmp, dX] = feval(fx, XNew, fxargs{:});
        dHdTheta = -( dX - 0.5*TraceInvGdG' + LastTerm' );
        
        ProposedMomentum = ProposedMomentum - TimeStep*(opt.StepSize/2)*dHdTheta;
        
        SavedSteps( StepNum + 1, : ) = XNew;
    end
    
    % Calculate proposed H value
    ProposedfX = feval(fx, XNew, fxargs{:});
       
    ProposedLogDet = 0.5*( log(2) + D*log(pi) + 2*sum(log(diag(CholG))) );
    
    ProposedH = -ProposedfX + ProposedLogDet + (ProposedMomentum'*InvG*ProposedMomentum)/2;
     
    % Calculate current H value
    CurrentLogDet = 0.5*( log(2) + D*log(pi) + 2*sum(log(diag(OriginalCholG))) );
    
    CurrentH  = -CurrentfX + CurrentLogDet + (OriginalMomentum'*OriginalInvG*OriginalMomentum)/2;
    
    % Accept according to ratio
    Ratio = -ProposedH + CurrentH;
    
    if Ratio > 0 || (Ratio > log(rand))
        CurrentfX = ProposedfX;
        X = XNew;
        Accepted = Accepted + 1;
        %disp('Accepted')
        XAccepted(Aidx,:) = X;
        Aidx = Aidx + 1;        
    else
        %disp('Rejected')
    end    
    
    % Save sample
    XSaved(IterationNum,:) = X; 
end

% remove trailing zeros
XAccepted = XAccepted(1:Aidx-1,:);

end