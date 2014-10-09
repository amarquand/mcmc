function [] = results(plot_variables)

try
    plot_variables;
catch
    plot_variables = false;
end

[statsname, working_dir] = uigetfile('','Select MCMC stats file');
[fname, fdir] = uigetfile('','Select chains for RMHMC variable');
[tname, tdir] = uigetfile('','Select chains for MH variable');

test_interval = 1;

load([working_dir,statsname]);

% Exclude burn-in samples
load([fdir,fname]);
load([tdir,tname]);
    
srange        = 6000:8000;%stats.opt.BurnIn:length(f_all);
f_post        = f_all(:,srange);
try
    LogTheta_post = LogTheta_all(:,srange);
catch
    LogTheta_post = Theta_all(:,srange);
end

f0       = zeros(size(f_post,1),1);
theta0 = zeros(size(LogTheta_post,1),1);

ESS_f     = results_ESS(f_post',length(f_post)-1);
ESS_theta = results_ESS((LogTheta_post)',length(LogTheta_post)-1);

N = length(f_post);
disp(['mean ESS (f) = ',num2str(100*mean(ESS_f)/N),'%']);
disp(['min ESS (f)  = ',num2str(100*min(ESS_f)/N),'%']);
disp(['mean ESS (theta) = ',num2str(100*mean(ESS_theta)/N),'%']);
disp(['min ESS (theta)  = ',num2str(100*min(ESS_theta)/N),'%']);

stats
if plot_variables
    results_generate_plots(f_post,f0);
    results_generate_plots((LogTheta_post),theta0);
end

[tmp, fmin] = min(ESS_f);
[tmp, tmin] = min(ESS_theta);
[tmp, fmax] = max(ESS_f);
[tmp, tmax] = max(ESS_theta);

m = median(ESS_f);
[tmp, fmed] = min(abs(ESS_f - m));

m = median(ESS_theta);
[tmp, tmed] = min(abs(ESS_theta - m));

fi = [fmin, fmed, fmax];
ti = [tmin, tmed, tmax];

figure
fthin = f_post(:,1:test_interval:end);
subplot(2,1,1)
plot(fthin(fi,1:min(length(fthin),1000))');
xlabel('iteration')
ylabel('f')
title('Trace: f')
%legend('min','med','max');
subplot(2,1,2)
% plot_acf_pretty(fthin(fmin,1:min(length(fthin),1000))')
% 
% figure
tthin = LogTheta_post(:,1:test_interval:end);
%subplot(2,1,1)
plot(tthin(ti,1:min(length(tthin),1000))');
xlabel('iteration')
ylabel('log(theta)')
title('Trace: log(theta)')
%legend('min','med','max');
%subplot(2,1,2)
%plot_acf_pretty(tthin(tmin,1:min(length(tthin),1000))')

end

