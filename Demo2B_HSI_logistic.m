%% Demo 2B: NS-KTR on HSI Data - Classification Tasks (Targets 5-8)
% This demo demonstrates the application of NS-KTR to hyperspectral imagery data
% for classification tasks (Targets 5-8): wheat cultivars (Heerup, Kvium, Rembrandt, Sheriff)

clc;
clear all;
close all;

% Set random seed for reproducibility
s = RandStream('mt19937ar', 'Seed', 2);
RandStream.setGlobalStream(s);

% Add necessary paths
addpath('external_methods/TensorReg-master/');
addpath('external_methods/tensor_toolbox-2.6/');
addpath('external_methods/AOAS21-QuantileTR/');

fprintf('Demo 2B: NS-KTR on HSI Data - Classification Tasks (Targets 5-8)\n');
fprintf('----------------------------------------------------------------\n');

%% Parameters
sampling_rate = 25;  % 25%, 50%, or 75%
target_index = 5;    % Target index (5-8)
                     % 5: Heerup (wheat cultivar)
                     % 6: Kvium (wheat cultivar)
                     % 7: Rembrandt (wheat cultivar)
                     % 8: Sheriff (wheat cultivar)
rank = 1;            % Tensor rank

% Target names for reference
target_names = {'Heerup', 'Kvium', 'Rembrandt', 'Sheriff'};
target_display = target_names{target_index-4}; % Adjust index for display

fprintf('Sampling rate: %d%%\n', sampling_rate);
fprintf('Target: %s (index %d)\n', target_display, target_index);
fprintf('Rank: %d\n\n', rank);

%% Load data
fprintf('Loading HSI data (sampling rate %d%%)...\n', sampling_rate);

try
    % Attempt to load data
    data_file = sprintf('data_leaf/Data_SR_%d.mat', sampling_rate);
    fprintf('  Loading %s...\n', data_file);
    Data = load(data_file);
    
    % Extract the specified target data
    X_train = tensor(Data.X_train_sampled);
    X_test = tensor(Data.X_test_sampled);
    X_val = tensor(Data.X_val_sampled);
    y_train = Data.y_train(:, target_index);
    y_test = Data.y_test(:, target_index);
    y_val = Data.y_val(:, target_index);
    
    % Convert continuous values to binary
    y_train = double(y_train > 0);
    y_test = double(y_test > 0);
    y_val = double(y_val > 0);
    
    fprintf('  Class distribution - Training: %.2f%% positive\n', 100*mean(y_train));
    fprintf('  Class distribution - Testing: %.2f%% positive\n', 100*mean(y_test));
    
    % Get dimensions
    I_full = size(X_train);
    I = I_full(1:end-1);
    D = ndims(X_train) - 1;
    N_train = size(X_train, D+1);
    N_test = size(X_test, D+1);
    N_val = size(X_val, D+1);
    
    fprintf('  Tensor dimensions: %s\n', mat2str(I));
    fprintf('  Number of dimensions: %d\n', D);
    fprintf('  Training samples: %d\n', N_train);
    fprintf('  Validation samples: %d\n', N_val);
    fprintf('  Test samples: %d\n\n', N_test);
    
catch ME
    % If data loading fails, create simulated data for demo purposes
    fprintf('  Error loading data: %s\n', ME.message);
end

%% Methods to compare
methods = {'KruskalTR', 'TuckerTR', 'NS-KTR-LS', 'NS-KTR-FL'};
method_labels = {'KruskalTR', 'TuckerTR', 'NS-KTR-LS', 'NS-KTR-FL'};
num_methods = length(methods);

fprintf('Methods to compare:\n');
for i = 1:num_methods
    fprintf('  %s\n', method_labels{i});
end
fprintf('\n');

%% First train the base LS model for initialization
fprintf('Training initialization model (NS-KTR-LS)...\n');
tic;
B_LS = KruskalTR_LS(I, D, X_train, y_train, rank);
ls_time = toc;

% Evaluate the LS model on the validation set
A_val = double(tenmat(X_val, D+1)) * khatrirao(B_LS.U(D:-1:1));
y_val_pred_ls = A_val * ones(rank,1) > 0;
val_acc_ls = mean(y_val_pred_ls == y_val);
fprintf('  Initialization model validation accuracy: %.4f\n\n', val_acc_ls);

%% ADMM options (for regularized methods)
opts = struct();
opts.rho = 1.0;
opts.tol = 1e-4;
opts.max_iter = 500;
opts.DEBUG = 0;

%% KruskalTR_reg options
opts2 = struct();
opts2.flag_warm = 1;  % Use warm start
opts2.tol_ALS = 1e-4;
opts2.t_max = 20;
opts2.numrep = 1;    % Use only one replica during parameter tuning
opts2.DEBUG = 0;

%% Evaluate all methods
fprintf('Evaluating all methods...\n\n');

% Initialize results storage
accuracy_results = zeros(num_methods, 1);
time_results = zeros(num_methods, 1);

for m_idx = 1:num_methods
    method = methods{m_idx};
    fprintf('  Evaluating method: %s\n', method);
    
    try
        tic;
        B_est = [];
        
        switch method
            case 'KruskalTR'
                % Kruskal Tensor Regression (Zhou et al.)
                regression_type = 'binomial';  % Logistic regression
                [~, B_est, ~] = kruskal_reg(zeros(N_train, 1), X_train, double(y_train), rank, regression_type);
                
            case 'TuckerTR'
                % Tucker Tensor Regression (Li et al.)
                regression_type = 'binomial';  % Logistic regression
                tucker_rank = repmat(rank, 1, D);
                [~, B_est, ~] = tucker_reg(zeros(N_train, 1), X_train, double(y_train), tucker_rank, regression_type);
                

            case 'NS-KTR-LS'
                % Our base method (no regularization)
                lambda1 = [0, 0, 0];   % L1 regularization
                lambda2 = [0, 0, 0];   % TV regularization
                lambda3 = [0, 0, 0];   % L2 regularization
                [B_est, ~] = KruskalTR_reg_logistic(I, D, X_train, y_train, rank, B_LS, lambda1, lambda2, lambda3, 'FL', opts2);
                
            case 'NS-KTR-FL'
                % Our method with Fused LASSO regularization and non-negativity
                lambda1 = [1e-1, 1e-1, 1e-1];   % L1 regularization
                lambda2 = [1e-1, 1e-1, 1e-1];   % TV regularization
                lambda3 = [1e-1, 1e-1, 1e-1];   % L2 regularization
                [B_est, ~] = KruskalTR_reg_logistic(I, D, X_train, y_train, rank, B_LS, lambda1, lambda2, lambda3, 'FL', opts2);
        end
        
        time_results(m_idx) = toc;
        
        % If model fitting successful, predict on test set
        if ~isempty(B_est)
            % Convert to double for prediction if needed
            if ~isa(B_est, 'ktensor')
                B_est_tensor = double(B_est);
            else
                B_est_tensor = B_est;
            end
            
            % Make predictions based on method type
            if strcmp(method, 'NS-KTR-LS') || strcmp(method, 'NS-KTR-FL')
                % For our methods that return ktensor
                A_test = double(tenmat(X_test, D+1)) * khatrirao(B_est.U(D:-1:1));
                scores = A_test * ones(rank,1);
                y_test_pred = scores > 0;
            else
                % For comparison methods
                if isa(B_est, 'ktensor')
                    A_test = double(tenmat(X_test, D+1)) * khatrirao(B_est.U(D:-1:1));
                    scores = A_test * ones(rank,1);
                else
                    A_test = double(tenmat(X_test, D+1));
                    scores = A_test * vec(B_est_tensor);
                end
                y_test_pred = scores > 0;
            end
            
            % Calculate accuracy
            accuracy = mean(y_test_pred == y_test);
            accuracy_results(m_idx) = accuracy;
            fprintf('    Test accuracy: %.4f, Time: %.2fs\n', accuracy, time_results(m_idx));
        else
            fprintf('    Model fitting failed\n');
            accuracy_results(m_idx) = NaN;
        end
        
    catch ME
        fprintf('    Error: %s\n', ME.message);
        accuracy_results(m_idx) = NaN;
        time_results(m_idx) = NaN;
    end
    
    % Clean up variables to free memory
    clear B_est B_est_tensor scores;
end

%% Visualize results
fprintf('\n--- Results comparison ---\n');
fprintf('Target: %s (Classification), Rank: %d\n\n', target_display, rank);

% Create results table
for m_idx = 1:num_methods
    fprintf('%-12s: Accuracy = %.4f, Time = %.2fs\n', ...
        method_labels{m_idx}, accuracy_results(m_idx), time_results(m_idx));
end

% Plot accuracy comparison
figure('Position', [100, 100, 800, 400]);

subplot(1, 2, 1);
bar(accuracy_results);
set(gca, 'XTick', 1:num_methods, 'XTickLabel', methods, 'XTickLabelRotation', 45);
ylabel('Classification Accuracy');
title(sprintf('%s Classification Accuracy (Rank = %d)', target_display, rank));
grid on;

subplot(1, 2, 2);
bar(time_results);
set(gca, 'XTick', 1:num_methods, 'XTickLabel', methods, 'XTickLabelRotation', 45);
ylabel('Computation Time (seconds)');
title('Computation Time Comparison');
grid on;

% Adjust figure for better display
fig = gcf;
fig.Position = [100, 100, 1000, 500];
sgtitle(sprintf('NS-KTR Performance on %s (Sampling Rate: %d%%)', ...
    target_display, sampling_rate));

fprintf('\nDemo completed successfully!\n');