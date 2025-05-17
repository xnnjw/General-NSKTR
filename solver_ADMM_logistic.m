function [x, misc] = solver_ADMM_logistic(A, y, R, x_init, lambda_1, lambda_2, lambda_3, opts)
    %
    % Solve the logistic regression with nonnegative fused Lasso (FL) using ADMM
    %                       Objective:
    % min_x sum(log(1 + exp(-y .* (A*x)))) + lambda_1*||x||_1 + lambda_2||Dx||_1 + lambda_3/2 * ||x||_2^2
    % 
    % subject to x >= 0 (if nonnegative constraint is needed)
    % -----------------------------------------------------------------------
    % Inputs:
    %   - A:         Design matrix (n x p)
    %   - y:         Response vector (n x 1) with values {-1, 1} for binary classification
    %   - R:         Number of groups for fused lasso regularization
    %   - x_init:    Initial value of x (p x 1)
    %   - lambda_1:  Regularization parameter for the L1 norm of x
    %   - lambda_2:  Regularization parameter for the fused lasso term
    %   - lambda_3:  Regularization parameter for the Ridge term
    %   - opts:      Structure with optional fields:
    %                - rho: Augmented Lagrangian parameter (default: 1.0)
    %                - tol: Convergence tolerance (default: 1e-4)
    %                - max_iter: Maximum number of iterations (default: 1000)
    %                - DEBUG: 0 or 1 for detailed logging
    %                - nonneg: 0 or 1 for nonnegativity constraint
    %
    % Outputs:
    %   - x: Solution vector
    %   - misc: Additional information, including residuals, objective value, and iteration logs.
    % -----------------------------------------------------------------------
    % By Xinjue Wang (w.xnj@outlook.com), with modifications
    %

    % Default parameters
    rho = 1.0;
    tol = 1e-4;
    max_iter = 1000;
    DEBUG = 0;
    nonneg = 0; % By default, do not enforce nonnegativity

    % Override parameters with opts if provided
    if exist('opts', 'var')
        if isfield(opts, 'rho'), rho = opts.rho; end
        if isfield(opts, 'tol'), tol = opts.tol; end
        if isfield(opts, 'max_iter'), max_iter = opts.max_iter; end
        if isfield(opts, 'DEBUG'), DEBUG = opts.DEBUG; end
        if isfield(opts, 'nonneg'), nonneg = opts.nonneg; end
    end

    % Initialize variables
    p = size(A, 2);
    x = x_init;
    z1 = x; % Initialize z1 with x_init for better convergence
    z2 = zeros(p - R, 1);
    u1 = zeros(p, 1);
    u2 = zeros(p - R, 1);

    % Construct differencing matrix D
    I = eye(p / R);
    D_shifted = circshift(I, -1);
    D = I - D_shifted;
    D = D(1:end-1, :);
    D = kron(eye(R), D);

    % Initialize storage for logs
    obj_values = zeros(max_iter, 1);
    residuals = zeros(max_iter, 1);

    for k = 1:max_iter
        x_old = x;

        % Update x with Newton's method
        x = update_x(A, y, u1, z1, u2, z2, D, rho, x, lambda_3, nonneg);

        % Update z1 and z2 using proximal operators
        if nonneg
            z1 = prox_nonneg_l1(x + u1 / rho, lambda_1 / rho);
        else
            z1 = prox_l1(x + u1 / rho, lambda_1 / rho);
        end
        z2 = prox_l1(D * x + u2 / rho, lambda_2 / rho);

        % Update dual variables u1 and u2
        u1 = u1 + rho * (x - z1);
        u2 = u2 + rho * (D * x - z2);

        % Compute residuals
        primal_res = norm(x - z1) + norm(D * x - z2);
        dual_res = rho * (norm(z1 - x_old) + norm(z2 - D * x_old));

        % Compute objective value (logistic loss + regularization)
        logistic_loss = sum(log(1 + exp(-y .* (A * x))));
        obj_values(k) = logistic_loss + lambda_1 * norm(x, 1) + ...
                        lambda_2 * norm(D * x, 1) + 0.5 * lambda_3 * norm(x)^2;

        % Log information
        if DEBUG && (k == 1 || mod(k, 10) == 0 || primal_res < tol)
            fprintf('Iter %d: Obj=%.3e, Pri=%.4e, Dul=%.4e \n', ...
                k, obj_values(k), primal_res, dual_res);
        end

        % Check convergence
        if max(primal_res, dual_res) < tol
            break;
        end
    end

    % Trim unused logs
    obj_values = obj_values(1:k);
    residuals = residuals(1:k);

    % Store additional information
    misc = struct();
    misc.obj_values = obj_values;
    misc.residuals = residuals;
    misc.iterations = k;
end

function x = update_x(A, b, u1, z1, u2, z2, D, rho, x0, lam3, nonneg)
    % solve the x update for logistic regression
    %   minimize [ logistic_loss(x) + (lam3/2)||x||^2 + (rho/2)||x - z1 + u1/rho||^2 + (rho/2)||D*x - z2 + u2/rho||^2 ]
    % via Newton's method.
    % ---------------------------
    
    % Default parameters for Newton's method
    alpha = 0.01;    % Line search parameter
    alpha = 0.1;    % Line search parameter
    beta = 0.5;      % Line search reduction factor
    TOLERANCE = 1e-5;
    MAX_ITER = 50;
    
    if nargin < 11
        nonneg = 0; % By default, no nonnegativity constraint
    end
    
    DTD = D'*D;
    % Problem dimensions
    [N, d] = size(A);
    I = eye(d);

    % Initial guess
    if exist('x0', 'var') && ~isempty(x0)
        x = x0;
    else
        x = zeros(d, 1);
    end

    % Define the objective function
    function [f, g, H] = compute_f_g_H(w)
        % Logistic loss
        Aw = A * w;
        exp_term = exp(-b .* Aw);
        log_term = log(1 + exp_term);
        f_logistic = sum(log_term);
        
        % Regularization terms
        f_ridge = lam3/2 * norm(w)^2;
        f_prox1 = rho/2 * norm(w - z1 + u1/rho)^2;
        f_prox2 = rho/2 * norm(D*w - z2 + u2/rho)^2;
        
        % Total objective
        f = f_logistic + f_ridge + f_prox1 + f_prox2;
        
        % Gradient
        sigmoid = 1 ./ (1 + exp_term);
        g_logistic = -A' * (b .* (1 - sigmoid));
        g_ridge = lam3 * w;
        g_prox1 = rho * (w - z1 + u1/rho);
        g_prox2 = rho * D' * (D*w - z2 + u2/rho);
        
        g = g_logistic + g_ridge + g_prox1 + g_prox2;
        
        % Hessian
        if nargout > 2
            W = diag(sigmoid .* (1 - sigmoid));
            H_logistic = A' * W * A;
            H_ridge = lam3 * I;
            H_prox1 = rho * I;
            H_prox2 = rho * DTD;
            
            H = H_logistic + H_ridge + H_prox1 + H_prox2;
        end
    end

    % Newton's method
    for iter = 1:MAX_ITER
        % Compute function, gradient, and Hessian
        [fx, g, H] = compute_f_g_H(x);
        
        % Compute Newton direction
        dx = -H \ g;
        
        % Compute Newton decrement for stopping criterion
        lambda_sq = -g' * dx;
        
        % Check for convergence
        % if sqrt(lambda_sq) < TOLERANCE
        %     break;
        % end
        if abs(lambda_sq) < TOLERANCE
            break;
        end

        % Backtracking line search
        t = 1;
        fx_new = compute_f_g_H(x + t*dx);
        while fx_new > fx - alpha*t*lambda_sq
            t = beta * t;
            if t < 1e-10
                break; % Avoid numerical issues
            end
            fx_new = compute_f_g_H(x + t*dx);
        end
        
        % Update x
        x_new = x + t*dx;
        
        % Apply nonnegativity constraint if required
        if nonneg
            x_new = max(0, x_new);
        end
        
        x = x_new;
    end
end

function z = prox_l1(v, lambda)
    % Proximal operator for L1 norm (soft thresholding)
    z = sign(v) .* max(0, abs(v) - lambda);
end

function z = prox_nonneg_l1(v, lambda)
    % Proximal operator for L1 norm with nonnegativity constraint
    z = max(0, v - lambda);
end