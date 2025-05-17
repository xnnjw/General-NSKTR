function [B_final, time_final] = KruskalTR_reg(I, D, X, y, rank, B_init, lam1, lam2, lam3, method, opts2)
    % Function to estimate Kruskal tensor using alternating least squares or regularized methods

    best_err = Inf;  % Variable to track the best (lowest) error
    time_final = 0;  % Initialize time_final
    prev_err = Inf;  % Initialize previous error
    B_final = [];    % Initialize B_final to store the best result
    
    flag_warm = 1;
    tol_ALS = 1e-4;      % Set a tolerance for convergence
    t_max = 20;
    numrep = 10;
    DEBUG = 0;

    % Override parameters with opts if provided
    if exist('opts2', 'var')
        if isfield(opts2, 'flag_warm'), flag_warm = opts2.flag_warm; end
        if isfield(opts2, 'tol'), tol_ALS = opts2.tol_ALS; end
        if isfield(opts2, 'max_iter'), t_max = opts2.t_max; end
        if isfield(opts2, 'numrep'), numrep = opts2.numrep; end
        if isfield(opts2, 'DEBUG'), DEBUG = opts2.DEBUG; end
    end

    % Loop over replicates
    for rep = 1:numrep
        % Start timing for the replicate
        tic;
        
        if flag_warm % warm start or not
            B_est = B_init;
        else 
            B_est = ktensor(arrayfun(@(j) 1 - 2 * rand(I(j), rank), 1:D, 'UniformOutput', false));
        end

        % Iterate over t_max iterations
        for t = 1:t_max
            % Loop over each dimension
            for d = 1:D
                
                % Formulate matrix A for the current dimension
                A = reshape(double(tenmat(X, [D + 1, d])) * khatrirao([B_est.U(D:-1:d+1); B_est.U(d-1:-1:1)]), size(y, 1), I(d) * rank);

                % Choose method for solving the update step
                if strcmp(method, 'FL') || strcmp(method, 'EN')
                    [beta_est, ~] = solver_ADMM(A, y, rank, B_est.U{d}(:), lam1(d), lam2(d), lam3(d));
                elseif strcmp(method, 'nFL') || strcmp(method, 'nEN')
                    [beta_est, ~] = solver_nnADMM(A, y, rank, B_est.U{d}(:), lam1(d), lam2(d), lam3(d));
                end

                % Update the factor matrix for the current dimension
                B_est.U{d} = reshape(beta_est, I(d), rank);

            end

            % Calculate current error
            curr_err = norm(double(B_est) - double(ktensor(B_est.U)), 'fro');
    
            % Log information
            if DEBUG
                res_check = norm(y - A*beta_est);
                fprintf('Rep %d, Iter %d: Res=%.5e\n', ...
                    rep, t, res_check);
            end

            % Check for convergence
            if norm(prev_err - curr_err) / norm(prev_err) < tol_ALS
                break;  % Stop iterations if the error change is small
            end
            % Update prev_err for next iteration
            prev_err = curr_err;

        end

        % Calculate the error after all dimensions have been updated
        curr_err = norm(double(B_est) - double(ktensor(B_est.U)), 'fro');

        % If this replicate gives a better result, store it
        if curr_err < best_err
            best_err = curr_err;
            B_final = B_est;        % Store the best B_est
            time_final = toc;       % Store the time taken for this replicate
        end
    end

end
