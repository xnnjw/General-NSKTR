function [beta0_final,beta_final,SIC,obj_final,dof] = ...
    ten_qreg(X,M,y,r,varargin)
% Tensor quantile regression
%
% Input
% X: n-by-p0 regular covariate matrix
% M: p1*p2*...*pd*n tensor covariate 
% y: n-by-1 response vector
% r: rank of CP tensor regression
%
% Parameter name pairs
% 'tau': specified quantile level, default is 0.5
% 'B0': initial value for tensor parameter
% 'Display': 'off' (default) or 'iter'
% 'MaxIter': maximum iteration, default is 100
% 'TolFun': tolerence in objective value, default is 1e-4
% 'Replicates': number of intitial values to try, default is 5
%
% Output
% beta0_final: regular coefficients for X
% beta_final: tensor coefficient for M
% SIC: Schwarz-type information criterion
% obj_final: check loss
% dof: degrees of freedom

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('M', @(x) isa(x,'tensor') || isnumeric(x));
argin.addRequired('y', @isnumeric);
argin.addRequired('r', @isnumeric);
argin.addParamValue('tau', 0.5, @(x) isnumeric(x) && x>0 && x<1);
argin.addParamValue('B0', [], @(x) isnumeric(x) || ...
    isa(x,'tensor') || isa(x,'ktensor') || isa(x,'ttensor'));
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off') || ...
    strcmp(x,'iter'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-4, @(x) isnumeric(x) && x>0);
argin.addParamValue('Replicates', 5, @(x) isnumeric(x) && x>0);
argin.parse(X,M,y,r,varargin{:});

tau = argin.Results.tau;
B0 = argin.Results.B0;
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
TolFun = argin.Results.TolFun;
Replicates = argin.Results.Replicates;

% check validity of rank r
if isempty(r)
    r = 1;
elseif r==0
    [beta0_final,obj_final] = rq_fnm(X, y, tau);
    beta_final = 0;
    return;
end

% check dimensions
if isempty(X)
    X = ones(size(M,ndims(M)),1);
end
[n,p0] = size(X);
d = ndims(M)-1;             % dimension of tensor variates
p = size(M);                % sizes tensor variates
if p(end)~=n
    error('dimension of M does not match that of X!');
end
if n<p0 || n<r*max(p(1:end-1))
    error('sample size n is not large enough to estimate all parameters!');
end

% convert M into a tensor T (if it's not)
TM = tensor(M);

% if space allowing, pre-compute mode-d matricization of TM
if strcmpi(computer,'PCWIN64') || strcmpi(computer,'PCWIN32')
    iswindows = true;
    % memory function is only available on windows !!!
    [dummy,sys] = memory; %#ok<ASGLU>
else
    iswindows = false;
end
% CAUTION: may cause out of memory on Linux
if ~iswindows || d*(8*prod(size(TM)))<.75*sys.PhysicalMemory.Available %#ok<PSIZE>
    Md = cell(d,1);
    for dd=1:d
        Md{dd} = double(tenmat(TM,[d+1,dd],[1:dd-1 dd+1:d]));
    end
end

% check user-supplied initial value
if ~isempty(B0)
    % ignore requested multiple initial values
    Replicates = 1;
    % check dimension
    if ndims(B0)~=d
        error('dimension of B0 does not match that of data!');
    end
    % turn B0 into a tensor (if it's not)
    if isnumeric(B0)
        B0 = tensor(B0);    
    end
    % resize to compatible dimension (if it's not)
    if any(size(B0)~=p(1:end-1))
        B0 = array_resize(B0, p);
    end
    % perform CP decomposition if it's not a ktensor of correct rank
    if isa(B0,'tensor') || isa(B0,'ttensor') || ...
            (isa(B0, 'ktensor') && size(B0.U{1},2)~=r)
        B0 = cp_als(B0, r, 'printitn', 0);        
    end
    % make sure B0.U is a 1-by-d cell tensor
    B0.U = reshape(B0.U, 1, d);
end

% pre-allocate variables
obj_final = inf;

% loop for various intial values
for rep=1:Replicates
    
    if ~isempty(B0)
        beta = B0;
    else
        % initialize tensor regression coefficients from uniform [-1,1]
        beta = ktensor(arrayfun(@(j) 1-2*rand(p(j),r), 1:d, ...
            'UniformOutput',false));
    end
    
    obj0 = inf;
    % main loop
    for iter=1:MaxIter
        % update coefficients for the regular covariates
        if (iter==1)
            eta = double(tenmat(TM,d+1)*tenmat(beta,1:d));
            % [beta0, obj0] = rq_fnm(X, y, tau);
        else
            eta = Xj*beta{d}(:);
        end
        [betatmp, objtmp] = rq_fnm([X, eta], y, tau);
        beta0 = betatmp(1:end-1);
        % stopping rule
        diffobj = objtmp-obj0;
        obj0 = objtmp;
        if (abs(diffobj)<TolFun*(abs(obj0)+1))
            break;
        end
        % update scale of tensor coefficients and standardize
        beta = arrange(beta*betatmp(end));
        for j=1:d
            beta.U{j} = bsxfun(@times,beta.U{j},(beta.lambda').^(1/d));
        end
        beta.lambda = ones(r,1);            
        
        % cyclic update of the tensor coefficients
        eta0 = X*beta0;
        for j=1:d
            if j==1
                cumkr = ones(1,r);
            end
            if (exist('Md','var'))
                if j==d
                    Xj = reshape(Md{j}*cumkr,n,p(j)*r);
                else
                    Xj = reshape(Md{j}*khatrirao([beta.U(d:-1:j+1),cumkr]),...
                        n,p(j)*r);
                end
            else
                if j==d
                    Xj = reshape(double(tenmat(TM,[d+1,j]))*cumkr, ...
                        n,p(j)*r);
                else
                    Xj = reshape(double(tenmat(TM,[d+1,j])) ...
                        *khatrirao({beta.U{d:-1:j+1},cumkr}),n,p(j)*r);
                end
            end
            [betatmp,dummy] = rq_fnm([Xj,eta0], y, tau); %#ok<ASGLU>
            beta{j} = reshape(betatmp(1:end-1),p(j),r);
            eta0 = eta0*betatmp(end);
            cumkr = khatrirao(beta{j},cumkr);
        end
    end
    
    % record if it has a smaller loss
    if obj0<obj_final
        beta0_final = beta0;
        beta_final = beta;
        obj_final = obj0;
    end
    
    if strcmpi(Display,'iter')
        disp(' ');
        disp(['replicate: ' num2str(rep)]);
        disp([' iterates: ' num2str(iter)]);
        disp([' loss: ' num2str(obj0)]);
        disp([' beta0: ' num2str(beta0')]);
    end
    
end

if d==2
    SIC = log(obj_final / n) + log(n) / (2*n) *(r*(p(1)+p(2)-r)+p0);
    dof = r*(p(1)+p(2)-r)+p0;
else
    SIC = log(obj_final / n) + log(n) / (2*n) * (r*(sum(p(1:end-1))-d+1)+p0);
    dof = r*(sum(p(1:end-1))-d+1)+p0;
end

end