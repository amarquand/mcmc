function [] = results_generate_plots(X,true_f,range)

nbins = 10;

try 
    range;
catch
    range = 1:size(X,1);
end
X = X(range,:);  

try 
    fval = true_f;
    str = 'g--';
catch
    fval = zeros(size(X,1),1);
    str = 'r--';
end
    
figure
plot(X');
title ('Variable trace')

figure
plot_mean_acf(X,50)

c = range(1);
for f = 1:floor(size(X,1)/12)
    figure
    for s = 1:12
        subplot(3,4,s);
        n = hist(X(c,:),nbins);
        m = max(n) + round(0.5*max(n));
        hist(X(c,:),nbins);
        hold on;
        plot(fval(c)*ones(m,1),1:m,str);
        title (['var ',num2str(c)]);
        c = c+1;
    end
end

rem =  mod(size(X,1),12);
if rem > 0
    figure
    for s = 1:rem
        subplot(3,4,s);
        n = hist(X(c,:),nbins);
        m = max(n) + round(0.5*max(n));
        hist(X(c,:),nbins);
        hold on;
        plot(fval(c)*ones(m,1),1:m,str);
        title (['var ',num2str(c)]);
        c = c + 1;
    end
end

end

function [] = plot_mean_acf(X, max_lag)
    n = size(X,1);
    t = size(X,2);
    
    mean_acf = zeros(1,t);
    for i = 1:n
        acf = xcorr(detrend(X(i,:),'constant'),'coeff');
        mean_acf = mean_acf + acf(t:end);
        if i == 1
           subplot(2,1,1)
           a = acf(t:end);
           stem(a(1:max_lag));
           title 'Variable 1';
        end
    
    end
    mean_acf = mean_acf / n;
    
    subplot(2,1,2)
    stem(mean_acf(1:max_lag));
    title 'Mean of all variables'
end
