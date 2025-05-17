function [B_final, time_final] = KruskalTR_LS(I, D, X, y, rank, opts2)
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Tensor regression with rank estimation using AIC, BIC, and BC
    % I: Dimensions of tensor
    % D: Number of dimensions
    % X: Design tensor
    % Z: Covariate matrix
    % y: Response vector
    % rank: Tensor rank
    % t_max: Maximum number of iterations
    % numrep: Number of replicates
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    best_err = Inf;  % Variable to track the best (lowest) error
    B_final = [];    % Initialize B_final to store the best result
    time_final = 0;  % Initialize time_final
    prev_err = Inf;  % Initialize previous error
    
    tol_ALS = 1e-4;      % Set a tolerance for convergence
    t_max = 10;
    numrep = 10;

    % Override parameters with opts if provided
    if exist('opts2', 'var')
        if isfield(opts2, 'tol'), tol_ALS = opts2.tol_ALS; end
        if isfield(opts2, 'max_iter'), t_max = opts2.t_max; end
        if isfield(opts2, 'numrep'), numrep = opts2.numrep; end
    end

    % Loop over replicates
    for rep = 1:numrep
        % Start timing for the replicate
        tic;

        % Initialize B_est and beta0_est for this replicate
        B_est = ktensor(arrayfun(@(j) 1 - 2 * rand(I(j), rank), 1:D, 'UniformOutput', false));
        % beta0_est = randn(size(Z, 2), 1);

        % Iterate over t_max iterations
        for t = 1:t_max
            % Loop over each dimension
            for d = 1:D
                A = reshape(double(tenmat(X, [D + 1, d])) * khatrirao([B_est.U(D:-1:d+1); B_est.U(d-1:-1:1)]), size(y, 1), I(d) * rank);
                R = chol(A' * A + 1e-4 * eye(size(A, 2))); % Least Squares via Cholesky decomposition
                beta_est = R \ (R' \ (A' * y));
                B_est.U{d} = reshape(beta_est, I(d), rank); % Update the factor matrix for the current dimension
            end

            curr_err = norm(double(B_est) - double(ktensor(B_est.U)), 'fro');

            % Check for convergence
            if abs(prev_err - curr_err) / prev_err < tol_ALS
                break;  % Stop iterations if the error change is small
            end
            
            % Update prev_err for next iteration
            prev_err = curr_err;
        end

        % Choose the best out of replicates based on error
        if curr_err < best_err
            best_err = curr_err;
            B_final = B_est;        % Store the best B_est
            time_final = toc;       % Store the time taken for this replicate
        end
    end
end