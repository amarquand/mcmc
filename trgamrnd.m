function [rn,i] = trgamrnd(cl,ch,a,b)
  
maxiter = 1000;

x0 = (a-1)/b; % mode of gampdf

for i = 1:maxiter
    x  = cl + (ch-cl)*rand;
    if cl <= x0 && x0 <= ch && a >= 1
        g = x^(a-1)*exp(-b*x) / (x0^(a-1)*exp(-(a-1)));
    elseif ch < x0 && a >=1
        g = x^(a-1)*exp(-b*x) / (ch^(a-1)*exp(-b*ch));
    elseif x0 < cl && a >= 1 || a < 1
        g = x^(a-1)*exp(-b*x) / (cl^(a-1)*exp(-b*cl));
    else
        error('I don''t know what to do');
    end
    
    if rand < g
        rn = x;
        break;
    end
end
if i == maxiter
    warning('maximum number of iterations reached');
    rn = NaN;    
end
    
end