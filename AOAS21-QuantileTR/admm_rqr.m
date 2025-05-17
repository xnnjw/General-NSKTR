function [b,z,stats] = admm_rqr(y,X,D,lambda,tau,varargin)
% ADMM Solver for regularized quantile regression
%
% [b,z,stats] = admm_rqr(y,x,lambda,tau,varargin) returns the minimum of
%  \sum_i rho_tau(y_i-x_i b) +  lambda *|D b|
% INPUT:
%   y - n-by-1 response vector
%   X - n-by-p covariate matrix
%   lambda - tuning parameters 
%   tau - quantile level
% OPTIONAL:
%   rho - penalty parameter, default: 1
%   MaxIter - max number of iterations in main loop, default: 1000
%   tol - tolerance of convergence, default: 1e-3
%   retol - relative tolerance, default: 1e-2
%   QUIET - whether display information, default: 1
%   penalty - penalty function, default: grouplasso2
%   b0 - initial value, default: []
%
% OUTPUT:
%   b - p-by-1 coefficient vector with sparse structure
%   z - auxiliary variable z = D b
%   stats - algorithmic statistics


% Parse inputs
argin = inputParser;
argin.addRequired('y', @isnumeric);
argin.addRequired('X', @isnumeric);
argin.addRequired('D', @isnumeric);
argin.addRequired('lambda', @(x) isnumeric(x) && x>=0);
argin.addRequired('tau', @(x) isnumeric(x) && all(0<x) && all(x<1));
argin.addParameter('rho', 1, @(x) isnumeric(x) && x>0); 
argin.addParameter('MaxIter', 1000, @(x) isnumeric(x) && x>0);
argin.addParameter('tol', 1e-3, @(x) isnumeric(x) && x>0);
argin.addParameter('retol', 1e-2, @(x) isnumeric(x) && x>0);
argin.addParameter('QUIET', 1, @(x) isnumeric(x) && x>=0);
argin.addParameter('b0', [], @(x) isnumeric(x) || isempty(x));
argin.parse(y,X,D,lambda,tau,varargin{:});
MaxIter = argin.Results.MaxIter;
rho = argin.Results.rho;
tol = argin.Results.tol;
retol = argin.Results.retol;
QUIET = argin.Results.QUIET;
b0 = argin.Results.b0;


[n,p] = size(X);
q = size(D, 1);

% Initialize
if (isempty(b0))
    b = zeros(p,1); 
else
    b = b0;
end

if lambda == 0
    [b, obj] = rq_fnm(X, y, tau);
    stats.iterations = 1;
    stats.objval = obj;
    stats.spar = nnz(b) / numel(b);
    return;
end

w = zeros(n, 1);
z = zeros(q, 1);
u = zeros(n, 1); % scaled U
v = zeros(q, 1); % scaled V

Xty = X' * y;
A = X' * X + D' * D;

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'prim norm', 'eps pri', 'dual norm', 'eps dual', 'objective', 'b norm');
end

for k = 1:MaxIter

    % Update z
    z_old = z;
    z = max(0, D*b + v - lambda/rho) - max(0,  - D*b - v - lambda/rho);

    % Update b
    b_old = b;
    b = A \ (X' * u - D' *v + Xty - X' * w + D' * z);

    % Update w
    w_old = w;
    xi = y - X*b + u - (tau-1/2)/rho;
    w = max(0, xi - 1/(2*rho)) - max(0,  - xi - 1/(2*rho));
    
    % Update u
    u = u - (w + X*b - y);
    
    % Update v
    v = v - (z - D*b);
    
    % Stopping rule   
    res_pri = sqrt(norm(w + X*b - y)^2 + norm(D*b - z)^2);
    res_dual = rho*sqrt(norm((w-w_old)'*X)^2 + norm((z-z_old)'*D)^2);
    eps_pri = sqrt(n) * tol + retol * max([norm(X*b), norm(w), norm(y),...
        norm(D*b), norm(z)]);
    eps_dual = sqrt(p) * tol + retol * sqrt(norm(rho*u'*X)^2 + norm(rho*v'*D)^2);
    res_b = norm(b - b_old) / (norm(b) + 0.001);

    eta = y - X*b;
    objval = sum(abs(eta)/2 + (tau - 0.5).*eta) + lambda * norm(z, 1);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\t%10.4f\n', k, ...
        res_pri, eps_pri, res_dual, eps_dual, objval, res_b);
    end
    if  (res_pri < eps_pri && res_dual < eps_dual) || k==MaxIter
        % fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\t%10.4f\n', k, ...
        % res_pri, eps_pri, res_dual, eps_dual, objval, res_b);
        break;
    end
end

% Collect algorithmic statistics
stats.iterations = k;
stats.objval = objval;
stats.norm = norm(z, 1);
stats.spar = nnz(z);


end