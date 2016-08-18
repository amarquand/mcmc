function [rn,i] = trinvgamrnd(cl,ch,a,b)
  
maxiter = 1000;

x0 = b/(a+1); % mode of invgampdf

for i = 1:maxiter
    x  = cl + (ch-cl)*rand;
    if cl <= x0 && x0 <= ch
        g = x^-(a+1)*exp(-b/x) / (x0^-(a+1)*exp(-b/x0));
    elseif ch < x0 
        g = x^-(a+1)*exp(-b/x) / (ch^-(a+1)*exp(-b/ch));
    elseif x0 < cl 
        g = x^-(a+1)*exp(-b/x) / (cl^-(a+1)*exp(-b/cl));
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