function [xnew, varargout] = slice_sample(x,f,b,varargin)

% b: a vector specifying an interval, a scalar specifying a width or a cell
%    array of the form: {@sampling_function [l,r] {arguments}}

M = 5;   % maximum width = M*b

% draw a level to define a slice. This assumes that f(x) = log(p(x)).
fx = feval(f, x, varargin{:});
y  = fx + log(rand);             % y = log(p(x)) - e, where e ~ exp(1)

if iscell(b)
    %l = -1/eps;
    %r = 1/eps;
    l = b{2}(1);
    r = b{2}(2);
    g = b{1};
    if length(b) > 2
        gargs = b{3};
    else
        gargs = {};
    end
else
    if length(b) == 1
        % step out to find a bound large enough to contain the slice
        l  = x - b*rand;   % left bound
        r  = l + b;        % right bound
        j  = floor(M*rand);
        k  = M - 1 - j;
        % left bound
        while  j > 0 && y < feval(f, l, varargin{:})
            l  = l - b;
            j  = j - 1;
        end
        % right bound
        while  k > 0 && y < feval(f, r, varargin{:})
            r  = r + b;
            k  = k - 1;
        end
    else
        % just use the bounds provided
        l = b(1);
        r = b(2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% shrink the bound to sample from the slice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ll = l; rr = r; vout = cell(nargout-1,1);
while true
    if iscell(b)
        xnew = feval(g,ll,rr,gargs{:});
    else        
        xnew = ll + rand*(rr-ll);
    end
    [fxnew, vout{:}] = feval(f, xnew, varargin{:}); 
    if y < fxnew
        varargout = vout;
        break
    end
    if xnew < x, ll = xnew; else rr = xnew; end
end
end
