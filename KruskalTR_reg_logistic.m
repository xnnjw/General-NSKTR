function [B_final, time_final] = KruskalTR_reg_logistic(I, D, X, y, rank, B_init, lam1, lam2, lam3, method, opts2)
    % Function to estimate Kruskal tensor using logistic regression with regularization
    %
    % Inputs:
    %   - I: Dimensions of tensor 
    %   - D: Number of dimensions
    %   - X: Tensor covariates (I_1 x I_2 x ... x I_D x N)
    %   - y: Binary response vector (N x 1) with values {0, 1}
    %   - rank: Tensor rank for Kruskal decomposition
    %   - B_init: Initial tensor for warm start
    %   - lam1, lam2, lam3: Regularization parameters for each dimension
    %   - method: Regularization method ('FL', 'nFL', 'EN', 'nEN')
    %   - opts2: Additional options
    %
    % Outputs:
    %   - B_final: Final estimated tensor
    %   - time_final: Computation time
    
    best_err = Inf;  % Variable to track the best (lowest) error
    time_final = 0;  % Initialize time_final
    prev_err = Inf;  % Initialize previous error
    B_final = [];    % Initialize B_final to store the best result
    
    flag_warm = 1;   % Use warm start by default
    tol_ALS = 1e-4;  % Set a tolerance for convergence
    t_max = 20;      % Maximum iterations
    numrep = 10;     % Number of replicates
    DEBUG = 0;       % Debug flag
    
    % Convert binary responses from {0,1} to {-1,1} for logistic regression
    y_transformed = 2*y - 1;
    % y_transformed = y;
    
    % Override parameters with opts if provided
    if exist('opts2', 'var')
        if isfield(opts2, 'flag_warm'), flag_warm = opts2.flag_warm; end
        if isfield(opts2, 'tol_ALS'), tol_ALS = opts2.tol_ALS; end
        if isfield(opts2, 't_max'), t_max = opts2.t_max; end
        if isfield(opts2, 'numrep'), numrep = opts2.numrep; end
        if isfield(opts2, 'DEBUG'), DEBUG = opts2.DEBUG; end
    end
    
    % Set nonnegative flag based on method
    if strcmp(method, 'nFL') || strcmp(method, 'nEN')
        nonneg = 1;
    else
        nonneg = 0;
    end
    
    % ADMM options
    opts = struct();
    opts.rho = 1.0;
    opts.tol = 1e-4;
    opts.max_iter = 500;
    opts.DEBUG = DEBUG;
    opts.nonneg = nonneg;
    
    % Loop over replicates
    for rep = 1:numrep
        % Start timing for the replicate
        tic;
        
        % Initialize B_est - warm start or random
        if flag_warm && ~isempty(B_init)
            B_est = B_init;
        else
            % Initialize with random values
            if nonneg
                % Non-negative initialization
                B_est = ktensor(arrayfun(@(j) rand(I(j), rank), 1:D, 'UniformOutput', false));
            else
                % Random initialization between -1 and 1
                B_est = ktensor(arrayfun(@(j) 1 - 2 * rand(I(j), rank), 1:D, 'UniformOutput', false));
            end
        end
        
        % Iterate over t_max iterations
        for t = 1:t_max
            % Initialize variables to track changes
            delta_norm = 0;
            
            % Loop over each dimension
            for d = 1:D
                % Store old factor matrix for tracking changes
                B_old = B_est.U{d};
                
                % Formulate matrix A for the current dimension using MTTKRP
                A = reshape(double(tenmat(X, [D + 1, d])) * khatrirao([B_est.U(D:-1:d+1); B_est.U(d-1:-1:1)]), size(y, 1), I(d) * rank);
                
                % Initialize factor matrix as a vector
                beta_vec = B_est.U{d}(:);
                
                % Solve using logistic ADMM
                [beta_est, misc] = solver_ADMM_logistic(A, y_transformed, rank, beta_vec, lam1(d), lam2(d), lam3(d), opts);
                
                % Update the factor matrix for the current dimension
                B_est.U{d} = reshape(beta_est, I(d), rank);
                
                % Calculate change in this factor matrix
                delta_norm = delta_norm + norm(B_old - B_est.U{d}, 'fro');
                
                % Log information if DEBUG is enabled
                if DEBUG
                    fprintf('Rep %d, Iter %d, Dim %d: Obj=%.5e, Iters=%d\n', ...
                        rep, t, d, misc.obj_values(end), misc.iterations);
                end
            end
            
            % Normalize the factor matrices to avoid scaling issues
            B_est = normalize(B_est);
            
            % Compute current tensor reconstruction error
            curr_err = norm(double(B_est) - double(ktensor(B_est.U)), 'fro');
            
            % Check for convergence
            if abs(prev_err - curr_err) / (prev_err + eps) < tol_ALS || delta_norm < tol_ALS
                if DEBUG
                    fprintf('Converged at iteration %d with error %.6f\n', t, curr_err);
                end
                break;  % Stop iterations if the error change is small
            end
            
            % Update prev_err for next iteration
            prev_err = curr_err;
            
            % Log information
            if DEBUG && mod(t, 5) == 0
                % Calculate prediction accuracy
                A_final = reshape(double(tenmat(X, [D + 1, D])) * khatrirao(B_est.U(D-1:-1:1)), size(y, 1), I(D) * rank);
                beta_final = B_est.U{D}(:);
                y_pred = A_final * beta_final > 0;  % Threshold at 0
                accuracy = mean(y_pred == (y_transformed > 0));
                
                fprintf('Rep %d, Iter %d: Error=%.6f, Accuracy=%.4f\n', ...
                    rep, t, curr_err, accuracy);
            end
        end
        
        % Store timing for this replicate
        time_rep = toc;
        
        % Calculate final error for ranking replicates
        curr_err = norm(double(B_est) - double(ktensor(B_est.U)), 'fro');
        
        % Calculate prediction accuracy
        A_final = reshape(double(tenmat(X, [D + 1, D])) * khatrirao(B_est.U(D-1:-1:1)), size(y, 1), I(D) * rank);
        beta_final = B_est.U{D}(:);
        logits = A_final * beta_final;
        y_prob = 1./(1 + exp(-logits)); % Convert to probabilities
        y_pred = y_prob > 0.5;          % Threshold at 0.5
        accuracy = mean(y_pred == y);
        
        if DEBUG
            fprintf('Replicate %d completed in %.2f seconds, final error: %.6f, accuracy: %.4f\n', ...
                rep, time_rep, curr_err, accuracy);
        end
        
        % If this replicate gives a better result, store it
        if curr_err < best_err
            best_err = curr_err;
            B_final = B_est;        % Store the best B_est
            time_final = time_rep;  % Store the time taken for this replicate
            
            if DEBUG
                fprintf('New best model found at replicate %d\n', rep);
            end
        end
    end
end

% Utility function to normalize ktensor object
function T = normalize(T)
    % Normalize all the columns of the factor matrices to unit length
    for n = 1:ndims(T)
        for r = 1:size(T.U{n}, 2)
            tmp = norm(T.U{n}(:,r));
            if tmp > 0
                T.U{n}(:,r) = T.U{n}(:,r) / tmp;
            end
        end
    end
end