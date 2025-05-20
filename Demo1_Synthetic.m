%% Demo 1: NS-KTR on Synthetic Tensor Signals
% This demo demonstrates the performance of NS-KTR method on synthetic tensor signals
% comparing different regularization approaches within the NS-KTR framework.

clc;
clear all;
close all;

% Set random seed for reproducibility
s = RandStream('mt19937ar', 'Seed', 2);
RandStream.setGlobalStream(s);

% Add necessary paths (adjust as needed for your environment)
addpath('tensor_toolbox-v3.6/');

%% Parameters
N = 1000;           % Sample size
rank = 2;          % Tensor rank
SNR = 20;          % Signal-to-noise ratio (in dB)

% Choose a signal type
signal_type = 'Gradient';  % Options: 'Gradient', 'Floor', 'Wave', 'Fading_Cross'

fprintf('Demo 1: NS-KTR on Synthetic Tensor Signals\n');
fprintf('------------------------------------------\n');
fprintf('Signal type: %s\n', signal_type);
fprintf('Sample size: %d\n', N);
fprintf('Rank: %d\n', rank);
fprintf('SNR: %d dB\n\n', SNR);

%% Load or create synthetic signal
switch signal_type
    case 'Gradient'
        try
            load('data/B_gradient.mat');
            B_db = B_gradient;
        catch
            % Create synthetic gradient if file not found
            [X, Y] = meshgrid(linspace(0, 1, 128), linspace(0, 1, 128));
            B_db = X;
        end
        
    case 'Floor'
        try
            load('data/B_floor.mat');
            B_db = B_floor;
        catch
            % Create synthetic floor pattern if file not found
            [X, Y] = meshgrid(linspace(-1, 1, 128), linspace(-1, 1, 128));
            B_db = (abs(X) < 0.5) & (abs(Y) < 0.5);
        end
        
    case 'Wave'
        try
            load('data/B_smooth_wave.mat');
            B_db = B_smooth_wave;
        catch
            % Create synthetic wave pattern if file not found
            [X, Y] = meshgrid(linspace(-3, 3, 128), linspace(-3, 3, 128));
            B_db = sin(2*X) .* cos(2*Y);
        end
        
    case 'Fading_Cross'
        try
            load('data/B_3D_fadingcross.mat');
            B_db = B;
        catch
            % Create synthetic 3D fading cross if file not found
            [X, Y, Z] = meshgrid(linspace(-1, 1, 32), linspace(-1, 1, 32), linspace(-1, 1, 32));
            B_db = exp(-2*(X.^2 + Y.^2 + Z.^2)) .* (abs(X) < 0.2 | abs(Y) < 0.2 | abs(Z) < 0.2);
        end
end

B_t = tensor(B_db);  % Convert to tensor
I = size(B_db);      % Tensor dimensions
D = length(I);       % Number of dimensions

fprintf('Original tensor size: [%s]\n\n', num2str(size(B_db)));

%% Define methods to compare
methods = {
    struct('name', 'LS',    'lam1', [],      'lam2', [],      'lam3', []),
    struct('name', 'EN',    'lam1', [10, 10], 'lam2', [0, 0], 'lam3', [100, 100]),
    struct('name', 'nEN',   'lam1', [10, 10], 'lam2', [0, 0], 'lam3', [100, 100]),
    struct('name', 'FL',    'lam1', [50, 16], 'lam2', [1000, 1000], 'lam3', [0, 0]),
    struct('name', 'nFL',   'lam1', [50, 16], 'lam2', [1000, 1000], 'lam3', [0, 0])
};

% For 3D data, extend regularization parameters
if D == 3
    for i = 1:length(methods)
        if ~isempty(methods{i}.lam1)
            methods{i}.lam1 = [methods{i}.lam1, methods{i}.lam1(end)];
            methods{i}.lam2 = [methods{i}.lam2, methods{i}.lam2(end)];
            methods{i}.lam3 = [methods{i}.lam3, methods{i}.lam3(end)];
        end
    end
end

num_methods = length(methods);

%% Generate data
fprintf('Generating synthetic data...\n');

% Design tensor
X = tensor(randn([size(B_db), N]));  
I_full = size(X);  % Full size of X
I = I_full(1:end-1);  % Sizes of each tensor dimension

% Generate response
mu = double(ttt(B_t, X, 1:D));  % Mean response
signal_power = mean(mu(:).^2);  % Calculate signal power
noise_power = signal_power / 10^(SNR / 10);  % Calculate noise power from SNR
y = mu + sqrt(noise_power) * randn(size(mu));  % Add Gaussian noise

fprintf('Data generation complete.\n\n');

%% Plot original signal
fprintf('Visualizing original signal...\n');

figure;
if D == 2
    subplot(2, 3, 1);
    imagesc(B_db);
    colormap('gray');
    axis equal; axis tight;
    title('Original Signal', 'FontSize', 12);
else
    % For 3D signal, show central slice
    subplot(2, 3, 1);
    mid_slice = ceil(size(B_db, 3)/2);
    imagesc(squeeze(B_db(:,:,mid_slice)));
    colormap('gray');
    axis equal; axis tight;
    title('Original Signal (Mid Slice)', 'FontSize', 12);
end

%% Run each method
fprintf('Running different methods...\n\n');

% Initialize results storage
all_errors = zeros(num_methods, 1);
B_all = cell(num_methods, 1);

for k = 1:num_methods
    method = methods{k}.name;
    lam1 = methods{k}.lam1;
    lam2 = methods{k}.lam2;
    lam3 = methods{k}.lam3;
    
    fprintf('Method: %s\n', method);
    
    tic;
    if k == 1
        % Run Least Squares method (baseline)
        [B_est, ~] = KruskalTR_LS(I, D, X, y, rank);
        B_LS = B_est;  % Save for initialization of other methods
    else
        % Run regularized methods
        opts2 = struct();
        opts2.flag_warm = 1;  % Use warm start
        opts2.t_max = 20;     % Maximum iterations

        % Run the regularized method
        [B_est, ~] = KruskalTR_reg(I, D, X, y, rank, B_LS, lam1, lam2, lam3, method, opts2);
    end
    time_taken = toc;
    
    % Convert to dense tensor for visualization and error calculation
    B_est_db = double(B_est);
    
    % Calculate relative error
    err_est = norm(B_est_db(:) - B_db(:), 'fro') / norm(B_db(:), 'fro');
    all_errors(k) = err_est;
    B_all{k} = B_est_db;
    
    fprintf('  Relative Error: %.4f\n', err_est);
    fprintf('  Time: %.2f seconds\n\n', time_taken);
    
    % Plot the result
    if D == 2
        subplot(2, 3, k+1);
        imagesc(B_est_db);
        colormap('gray');
        axis equal; axis tight;
        title(sprintf('%s: Error=%.4f', method, err_est), 'FontSize', 12);
    else
        % For 3D signal, show central slice
        subplot(2, 3, k+1);
        mid_slice = ceil(size(B_est_db, 3)/2);
        imagesc(squeeze(B_est_db(:,:,mid_slice)));
        colormap('gray');
        axis equal; axis tight;
        title(sprintf('%s: Error=%.4f', method, err_est), 'FontSize', 12);
    end
end

%% Summarize results
fprintf('\n----- Results Summary -----\n');
fprintf('Signal: %s\n', signal_type);
fprintf('Rank: %d, SNR: %d dB, Samples: %d\n\n', rank, SNR, N);

% Print a table of results
fprintf('Method      Error\n');
fprintf('----------------\n');
for k = 1:num_methods
    fprintf('%-10s  %.4f\n', methods{k}.name, all_errors(k));
end
fprintf('\n');

fprintf('Demo completed successfully!\n');