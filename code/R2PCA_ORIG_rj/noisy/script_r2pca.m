
%% configure paths

clear all; close all; warning ('off','all'); % clc;
rand('state',sum(100*clock));

% if ~isdeployed
%     addpath('/Users/robertjones/Desktop/W23/553/project/R2PCA/noiseless/utils');
%     addpath('/Users/robertjones/Desktop/W23/553/project');
% end

% ===========================  ============================
%% Set parameters

% INPUT:   
%   d,N = size of the matrix is d x N
%   r = rank of low-rank component.
%   p = Probability of nonzero entry in sparse component
%   v = variance of the sparse entries
%   mu = coherence parameter of low-rank component
%   T = number of trials
%   k = size of expanded block/window for noisy-case
%   NoiseLvl = noise level of AGWN

d = 100;            % Size of the matrix = d x N
N = d;
r = 5;              % rank of low-rank component
p = 0.05;           % probability of sparse outliers
mu = 8;             % Coherence: value in [1,d/r]
v = 10;             % variance of the sparse entries %v=10 used in paper
k = 10;             % Size of block in noisy-case (k>r)
sigma_n = 1e-3;     % Noise level (standard deviation (sigma) of Gauss dist)

UseNoisyData = true; % Add noise to data or not?

verbose = 0;        % set =1 to display messages, =0 to not

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

if size(M,1)>size(M,2)
    %Transpose data matrix M so that size(M,1)<size(M,2)
    M = M.';
end

%% Run R2PCA recon
% ================== Run R2PCA ==================
fprintf('Running R2PCA...\n');

% make sure k>r, error if not
if k<=r, error('Must have k>r'); end

tic1 = tic;
[Lhat, Shat, Uhat, Chat, info] = nR2PCA(M, r, k, sigma_n, false, 1);
elaptime = toc(tic1);

% fprintf('=== LoR CRITERIA: for i=1:d-r ===\n');
% fprintf('=== LoR CRITERIA: while rank(auxA)<d-r ===\n');
fprintf('=== LoR CRITERIA: while rank(A)<d-r ===\n');

fprintf('Elap time = %g s\n',elaptime);

%Error in L
stats.err.l = norm(L-Lhat,'fro')/norm(L,'fro');
fprintf('L Error = %g \n',stats.err.l);

%Error in S
% (take abs val of Shat, zero any entries <1e-9)
Shat_ = abs(Shat);
Shat_(Shat_<sigma_n)=0;
stats.err.s = norm(S-Shat_,'fro')/norm(S,'fro');
fprintf('S Error = %g \n',stats.err.s);

% (binarize + compute dice score)
stats.dice.s = dice(S>0,Shat_>0);
fprintf('S Dice coeff = %g \n',stats.dice.s);


%% Make figures of recon results

f2 = figure('color','w','position',[289 205 990 661],'InvertHardcopy','off');
subplot(3,2,1:2);
imagesc(M); clim([0 1]); set(gca,'colormap',gray);
title('M = L + S'); set(gca,'FontSize',15);

subplot(323);
imagesc(L); clim([0 1]); set(gca,'colormap',gray);
title('L'); set(gca,'FontSize',15);
subplot(325);
imagesc(abs(Lhat)); clim([0 1]); set(gca,'colormap',gray);
title('L hat'); set(gca,'FontSize',15);

subplot(324);
imagesc(S); clim([0 1]); set(gca,'colormap',gray);
title('S'); set(gca,'FontSize',15);
subplot(326);
imagesc(abs(Shat)); clim([0 1]); set(gca,'colormap',gray);
title('S hat'); set(gca,'FontSize',15);



