function [ Lhat, Uhat, Shat, Coeffs, elaptime ] = run_r2pca(d, N, r, p, mu, v, k, sigma_n, UseNoisyData, verbose)
% 
%function [ ] = run_r2pca(d, N, r, p, mu, v, k, sigma_n[, UseNoisyData, verbose])
% 
% INPUT:   
%   d,N = size of the matrix is d x N
%   r = rank of low-rank component.
%   p = Prob of nonzero entry in sparse component (prob of sparse outlier)
%   v = variance of the sparse entries
%   mu = coherence parameter of low-rank component (value in [1,d/r])
%   T = number of trials
%   k = size of expanded block/window for noisy-case
%   (opt.) sigma_n = noise level (sigma, std dev) of Gaussian additive noise
%   (opt.) UseNoistData = true to add noise (only supports true right now...)


%% configure paths

clear all; close all; warning ('off','all'); % clc;
rand('state',sum(100*clock));

% ===========================  ============================
fprintf('\n---- ---- ---- --------------- ---- ---- ---- \n')
fprintf('---- ---- ---- RPCA PARAMETERS ---- ---- ---- \n')
fprintf('-Matrix size (d x N):        \t d = %d, N = %d\n',d,N);
fprintf('-Low-rank dim:               \t r = %d \n',r);
fprintf('-Prob of sp outliers:        \t p = %g \n',p);
fprintf('-Var. or sparse outliers:    \t v = %g \n',v);
fprintf('-Coherence:                  \t %s = %g \n',char(956),mu);
fprintf('-R2PCA noisy search wind sz: \t k = %d \n',k);
fprintf('-Gauss. noise std dev:       \t %s_noise = %g \n',char(963),sigma_n);
fprintf('-Verbose:                    \t verbose = %d \n',verbose);
fprintf('---- ---- ---- --------------- ---- ---- ---- \n\n')


% ===========================  ============================
%% Generate synthetic/simulated data
fprintf('\n======== GENERATING SIMULATED DATA ==========\n');

s = ceil(p*N);               %number of corrupted entries per row
fprintf(' -# of corrupted entries per row = %g \n',s);

if s > (N-r)/(2*(r+1))
    warning('---The # of corrupted entries in S is large! (S not sparse enough)---');
end

% ===================== GENERATE U MATRIX ============================
% ======= Low-rank basis within range of coherence parameter mu =======
fprintf(' -Generating low-rank subspace U with coherence ~%s=%g \n',char(956),mu);
U = basisWcoherence(d,r,mu-.5);
while coherence(U)>mu+.5
    U = basisWcoherence(d,r,mu-.5);
end
U = orth(U);
mu = coherence(U);
fprintf('\t -Final coherence value: %s = %g \n',char(956),mu);

% ===================== GENERATE L MATRIX (& THETA) ============================
fprintf(' -Generating low-rank matrix L\n');
fprintf('\t L = U*%s \n',char(920));
fprintf('\t %s = random coeffs of low-rank component\n',char(920));
Theta = randn(r,N);     % Coefficients of low-rank component
L = U*Theta;            % Low-rank component

% ===================== GENERATE S MATRIX ============================
% ========== Sparse matrix with s corrupted entries per row ==========
fprintf(' -Generating sparse outlier matrix S\n');
fprintf(' \t with %g corrupted entries per row \n',s);
% randomly generate S matrix w/ specified # of outliers in each row
S = zeros(d,N);
for i = 1:d
    S(i,randsample(N,s)) = v*randn(s,1);
end
% Verify that each column has at least r+1 uncorrupted entries
for j=1:N
    idx = find(S(:,j));
    nonZeros = length(idx);
    if nonZeros>d-r-1
        newZeros = nonZeros-(d-r-1);
        S(idx(randsample(nonZeros,newZeros)),j) = 0;
    end
end

% ===================== GENERATE W (NOISE) MATRIX ============================
% =============== Construct synthetic additive noise matrix ==========
fprintf(' -Generating Gauss noise matrix W\n');
fprintf(' \t with %s_noise = %g \n',char(963),sigma_n);

W = normrnd(0,sigma_n^2,[d,N]);

% ===================== GENERATE M (DATA/OBS) MATRIX ============================
% ======================= Mixed matrix =======================
fprintf(' -Creating observation data matrix M \n');
fprintf(' \t via M = L + S + W \n');
if UseNoisyData
    fprintf('--USING NOISY DATA--\n');
    M = L + S + W;
else
    fprintf('--USING NOISE-FREE DATA--\n');
    M = L + S;
end


%% Run R2PCA recon
% ================== Run R2PCA ==================
fprintf('Running R2PCA...');
tic1 = tic;

% make sure k>r, error if not
if k<=r, error('Must have k>r'); end

% 1st part of R2PCA: recover basis of low-rank component
Uhat = LoR_noisy(M, r, k, sigma_n, verbose);        
% 2nd part of R2PCA: recover coefficients
Coeffs = Sp_noisy(M, Uhat, k, sigma_n, r, verbose);  
% Now recover low rank component
Lhat = Uhat * Coeffs;        % Recover low-rank compoment
% And recover sparse component
Shat = M - Lhat;

elaptime = toc(tic1);
Lerr = norm(L-Lhat,'fro')/norm(L,'fro');

if verbose>0
    fprintf('==== FINISHED RECON ====\n');
    fprintf('\tTotal elapsed time:\t%g seconds\n',elaptime);
    fprintf('\tL reconst. error:  \t%g \n',Lerr);
    fprintf('=================(parameters:)======================\n');
    fprintf('d=%d\tN=%d\tr=%g\tp=%g\nmu=%g\tv=%g\tk=%g\tsigman=%g\n', d,N,r,p,mu,v,k,sigma_n);
    fprintf('==================================================\n');
end



