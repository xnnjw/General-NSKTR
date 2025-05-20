%% Demo
clear;
% addpath 'tensor_toolbox_2.6'
% addpath 'TensorReg'
% addpath 'SparseReg'

%%
n = 1000;
signal = 'One_brick';
% signal = 'Two_bricks';
% signal = '3D_Cross';
% signal = 'Pyramid';
% dist = 'normal';
dist = 'cauchy';
sigma = 1;
nsims = 5;
prop = 0.8;

%% basic settings
% True coefficients for regular covariates
p0 = 5;
b0 = ones(p0,1);
n1 = round(n * prop);
tau = 0.5;
    
%% shape of signal
[b, r] = generate_signal(signal, 30, 30, 30, n);
[p1,p2,p3] = size(b);

%% collectors
Y = NaN(n, nsims);
Mu = NaN(n, nsims);
Q = NaN(n, length(tau), nsims);

if ~strcmp(dist, 'normal')
    load([sprintf('cauchy%d_100_%d',n, sigma),'.mat']);  
end

beta0 = NaN(p1, p2, p3, length(tau), nsims);

predQ1 = NaN(n, nsims);
beta1 = NaN(p1, p2, p3, nsims);
predQ2 = NaN(n, length(tau), nsims);
beta2 = NaN(p1, p2, p3, length(tau), nsims);
predQ3 = NaN(n, length(tau), nsims);
beta3 = NaN(p1, p2, p3, length(tau), nsims);


mae1 = NaN(nsims, 1); % TR
mae2 = NaN(nsims, length(tau)); % TQR
mae3 = NaN(nsims, length(tau)); % RTQR IV

mse1 = NaN(nsims, 1); % linear regression
mse2 = NaN(nsims, length(tau)); % TQR
mse3 = NaN(nsims, length(tau)); % RTQR IV

%% Replications
for isim = 1:nsims
    
    % isim = 1;
    % reset random seed
    rng(100182020+isim);

    %%
    % Simulate covariates
    X = randn(n,p0);   % n-by-p regular design matrix
    M = tensor(randn(p1,p2,p3,n));  % p1-by-p2-by-n matrix variates

    %% Response
    mu = X*b0 + double(ttt(tensor(b), M, 1:3));

    if strcmp(dist, 'normal')
        error = randn(n, 1) * sigma;
    else
        error = error_all(isim, :)';
    end
    
    y = mu + error;
    
    Mu(:, isim) = mu;
    Y(:, isim) = y;

    for j = 1:length(tau)
        if strcmp(dist, 'normal')
            Q(:, j, isim) = norminv(tau(j), 0, sigma) + mu;
        else
            Q(:, j, isim) = Q_all(:,j,isim) + mu;
        end
    end
    
    idx1 = 1:n1;
    idx2 = n1+1:n;
    y1 = y(idx1);
    X1 = X(idx1, :);
    M1 = M(:,:,:,idx1);
    m_idx = find(tau==0.5,1);
    
   %% TR
   [beta01tmp,beta1tmp,~,~] = kruskal_reg(X1,M1,y1,r,'normal');
   beta1(:,:,:,isim) = beta1tmp;
   predQ1(:,isim) = X * beta01tmp + double(ttt(tensor(beta1tmp), tensor(M), 1:ndims(M)-1));
   mse1(isim, 1) = mean(mean(mean((double(beta1(:,:,:,isim)) - b).^2)));
   mae1(isim, 1) = mean(abs(predQ1(idx2,isim) - Q(idx2,m_idx,isim)));

   %% TQR
   for j = 1:length(tau)
       [beta02tmp, beta2tmp, BIC2, obj2, dof2] = ten_qreg(X1,M1,y1,r,'tau',tau(j));
       beta2(:,:,:,j,isim) = beta2tmp;
       predQ2(:,j,isim) = X * beta02tmp + double(ttt(tensor(beta2tmp), tensor(M), 1:ndims(M)-1));
       mse2(isim, j) = mean(mean(mean((double(beta2(:,:,:,j,isim)) - b).^2)));
       mae2(isim, j) = mean(abs(predQ2(idx2,j,isim) - Q(idx2,j,isim)));
   end

    %% RTQR IV
   Lam = exp(-2:3);
   for j = 1:length(tau)
       
        BIC3 = Inf(length(Lam),1);
        beta03_all = zeros(p0,length(Lam));
        beta3_all = zeros(p1,p2,p3,length(Lam));
        
        t_start = tic;
        for k = 1:length(Lam)
            [beta03tmp, beta3tmp, BIC3tmp, obj3, dof3tmp] = ten_regqreg(X1,M1,...
                y1,r,Lam(k),'tau', tau(j), 'pentype', 'fuse',...
                'B0', beta2(:,:,:,j,isim),'Display','iter');
            BIC3(k) = BIC3tmp;      
            beta03_all(:,k) = beta03tmp;
            beta3_all(:,:,:,k) = beta3tmp;
        end
        toc(t_start);
        
        sel3 = find(BIC3==min(BIC3),1);
        beta03tmp = beta03_all(:,sel3);
        beta3tmp = beta3_all(:,:,:,sel3);
        beta3(:,:,:,j,isim) = beta3tmp;
        predQ3(:,j,isim) = X * beta03tmp + double(ttt(tensor(beta3tmp), tensor(M), 1:ndims(M)-1));
        mse3(isim, j) = mean(mean(mean((double(beta3(:,:,:,j,isim)) - b).^2)));
        mae3(isim, j) = mean(abs(predQ3(idx2,j,isim) - Q(idx2,j,isim)));      
   end
end

%% Summarize results
beta1_re = median(beta1, 4);

beta2_re = reshape(beta2(:,:,:,m_idx,:),p1,p2,p3,nsims);
beta2_re = median(beta2_re, 4);

beta3_re = reshape(beta3(:,:,:,m_idx,:),p1,p2,p3,nsims);
beta3_re = median(beta3_re, 4);

disp(['TR MSE: ', num2str(median(mse1))])
disp(['TR MAE: ', num2str(median(mae1))])

disp(['TQR MSE: ', num2str(median(mse2))])
disp(['TQR MAE: ', num2str(median(mae2))])

disp(['RTQR MSE: ', num2str(median(mse3))])
disp(['RTQR MAE: ', num2str(median(mae3))])

%% Display 3D tensor
mymap = double(hot);
mymap = mymap(size(mymap,1):-1:1,:);

%1 brick & 2 bricks
if strcmpi(signal, 'One_brick') || strcmpi(signal, 'Two_bricks')
    xlim=[14 25];
    ylim=[14 25];
    zlim=[19 30];
elseif strcmpi(signal, '3D_Cross')%3D Cross
    xlim=[14 29];
    ylim=[14 29];
    zlim=[14 29];
elseif strcmpi(signal, 'Pyramid')% Pyramid
    xlim=[14 30];
    ylim=[14 30];
    zlim=[19 24];
end
    
pointsize = 10; %6;
cr = 1; %color limit range
 
figure; hold on;
set(gca,'FontSize',20);

subplot(2,2,1);
idx = find(b);
[XX, YY, ZZ] = ind2sub(size(b), idx);
scatter3(XX(:), YY(:), ZZ(:), pointsize, b(idx));
set(gca,'XLim',xlim,'YLim',ylim,'ZLim',zlim) %,'FontSize',22  for 3D Cross
title({'True Signal'}); %,'FontSize',22  for 3D Cross
colormap(mymap);
% colorbar;
caxis([0 cr]);


subplot(2,2,2);
idx = find(b);
[XX, YY, ZZ] = ind2sub(size(beta1_re), idx);
scatter3(XX(:), YY(:), ZZ(:), pointsize, beta1_re(idx));
set(gca,'XLim',xlim,'YLim',ylim,'ZLim',zlim) %,'FontSize',22  for 3D Cross
title({'TR'}); %,'FontSize',22  for 3D Cross
colormap(mymap);
% colorbar;
caxis([0 cr]);

subplot(2,2,3);
idx = find(beta2_re);
[XX, YY, ZZ] = ind2sub(size(beta2_re), idx);
scatter3(XX(:), YY(:), ZZ(:), pointsize, beta2_re(idx));
set(gca,'XLim',xlim,'YLim',ylim,'ZLim',zlim)
title({'TQR'});
colormap(mymap);
% colorbar;
caxis([0 cr]);

subplot(2,2,4);
idx = find(beta3_re);
[XX, YY, ZZ] = ind2sub(size(beta3_re), idx);
scatter3(XX(:), YY(:), ZZ(:), pointsize, beta3_re(idx));
set(gca,'XLim',xlim,'YLim',ylim,'ZLim',zlim)
title({'RTQR'});
colormap(mymap);
% colorbar;
caxis([0 cr]);
 
colorbar;
hp4 = get(subplot(2,2,4),'Position');
colorbar('Position', [hp4(1)+hp4(3)+0.022  hp4(2)  0.03  hp4(2)+hp4(3)*2.1],...
    'FontSize',9.5) 
%'Position' [left-right, up-down, width, height?]

