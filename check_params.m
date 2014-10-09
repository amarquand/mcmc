function p = check_params(opt)

if ~isfield(opt,'X0_RMHMC') || ~isfield(opt,'X0_MH')
    error('starting estimates opt.XO_RMHMC and opt.XO_MH must be defined');
end

% fill in generic defaults
def.nGibbsIter     = 10000;    
def.WriteInterim   = true; 
%def.BurnIn         = 1000;
def.PriorParam     = [2,2];
def.OptimiseTheta  = false;
def.CovFunc        = 'covfunc_sum'; %'covfunc_wsum';
def.OutputFilename = [];            % leave blank for no output
def.UseRMHMC       = false;   
def.UseGMassForHMC = false;            % Only used if UseRMHMC = false
% set parameters for testing ...
def.TestInterval = 2;      % factor to thin Markov chain by
def.nTestSamples = 1;      % How many samples of f to draw from N(mu_s,S_s)
flds = fields(def);
for i = 1:length(flds)
    fld = flds(i);
    if ~isfield(opt,fld)
        warning(['option ',char(fld),' not specified. Using default value']);
        eval(['opt.',char(fld),'=def.',char(fld),';']);
    end
end

% fill in defaults for RMHMC
def.rmhmc.NumOfIterations    = 1; 
def.rmhmc.StepSize           = 0.5;
def.rmhmc.NumOfLeapFrogSteps = 10;
def.rmhmc.NumOfNewtonSteps   = 4;
flds = fields(def.rmhmc);
for i = 1:length(flds)
    fld = flds(i);
    if ~isfield(opt.rmhmc,char(fld))
        warning(['option ',char(fld),' not specified for RMHMC. Using default value']);
        eval(['opt.rmhmc.',char(fld),'=def.rmhmc.',char(fld),';']);
    end
end

% fill in defaults for MH
def.mh.StepSize           = 0.2; 
flds = fields(def.mh);
for i = 1:length(flds)
    fld = flds(i);
    if ~isfield(opt.mh,char(fld))
        warning(['option ',char(fld),' not specified for MH. Using default value']);
        eval(['opt.mh.',char(fld),'=def.mh.',char(fld),';']);
    end
end

p = opt;
end

