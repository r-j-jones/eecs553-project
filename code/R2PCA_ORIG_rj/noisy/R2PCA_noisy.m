function [ Lhat, Uhat, Shat, Coeffs, elaptime ] = R2PCA_noisy( M, r, k, sigma_n )

% ================== Run R2PCA ==================
fprintf('Running R2PCA...');
tic1 = tic;

% make sure k>r, error if not
if k<=r, error('Must have k>r'); end

% 1st part of R2PCA: recover basis of low-rank component
Uhat = LoR_noisy(M, r, k, sigma_n );   

% 2nd part of R2PCA: recover coefficients
Coeffs = Sp_noisy(M, Uhat, k, sigma_n, r );  

% Now recover low rank component
Lhat = Uhat * Coeffs;        % Recover low-rank compoment

% And recover sparse component
Shat = M - Lhat;

%total recon time
elaptime = toc(tic1);

end