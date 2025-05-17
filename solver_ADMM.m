function [x, misc] = solver_ADMM(A, y, R, x_init, lambda_1, lambda_2, lambda_3, opts)
    %
    % Solve the nonnegative fused Lasso (FL) problem using the ADMM algorithm
    %                       Objective:
    % min_x 0.5||y - Ax||_2^2 + lambda_1*||x||_1 + lambda_2||Dx||_1 + lambda_3/2 * ||x||_2^2
    % -----------------------------------------------------------------------
    % Inputs:
    %   - A:         Design matrix (n x p)
    %   - y:         Response vector (n x 1)
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
    %
    % Outputs:
    %   - x: Solution vector
    %   - misc: Additional information, including residuals, objective value, and iteration logs.
    % -----------------------------------------------------------------------
    % version 2.0 - Updated to handle Ridge term
    % By Xinjue Wang (w.xnj@outlook.com)
    %

    % Default parameters
    % rho = norm(A,"fro")^2/(size(A,2));
    rho =1;
    tol = 1e-5;
    max_iter = 500;
    DEBUG = 0;

    % Override parameters with opts if provided
    if exist('opts', 'var')
        if isfield(opts, 'rho'), rho = opts.rho; end
        if isfield(opts, 'tol'), tol = opts.tol; end
        if isfield(opts, 'max_iter'), max_iter = opts.max_iter; end
        if isfield(opts, 'DEBUG'), DEBUG = opts.DEBUG; end
    end

    % Initialize variables
    p = size(A, 2);
    x = x_init;
    z1 = randn(p, 1);
    z2 = randn(p - R, 1);
    u1 = randn(p, 1);
    u2 = randn(p - R, 1);

    % Construct differencing matrix D
    I = eye(p / R);
    D_shifted = circshift(I, -1);
    D = I - D_shifted;
    D = D(1:end-1, :);
    D = kron(eye(R), D);

    % Precompute matrix inverses
    AtA = A' * A;
    ATy = A' * y;
    M = AtA + (rho + lambda_3) * eye(p) + rho * (D' * D);
    invM = M \ eye(p);

    % Initialize storage for logs
    obj_values = zeros(max_iter, 1);
    residuals = zeros(max_iter, 1);

    for k = 1:max_iter
        x_old = x;

        % Update x
        b = ATy + rho * (z1 - u1 / rho) + rho * (D' * (z2 - u2 / rho));
        x = invM * b;

        % Update z1 and z2 using proximal operators
        z1 = prox_l1(x + u1 / rho, lambda_1 / rho);
        z2 = prox_l1(D * x + u2 / rho, lambda_2 / rho);

        % Update dual variables u1 and u2
        u1 = u1 + rho * (x - z1);
        u2 = u2 + rho * (D * x - z2);

        % Compute residuals
        primal_res = norm(x - z1) + norm(D * x - z2);
        dual_res = rho * (norm(z1 - x_old) + norm(z2 - D * x_old));

        % Compute objective value
        obj_values(k) = 0.5 * norm(A * x - y)^2 + lambda_1 * norm(x, 1) + ...
                        lambda_2 * norm(D * x, 1) + 0.5 * lambda_3 * norm(x, 2)^2;

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

function z = prox_l1(v, lambda)
    % Proximal operator for L1 norm (soft thresholding)
    z = max(0, v - lambda) - max(0, -v - lambda);
end
