function [ Coeffs ] = Sp_noisy(M, U, k, noiselevel, r, verbose)

% function Coeffs = Sp_noisy(M, U, k, noiselevel, r[, verbose])
%  
%   Recover coefficients (in subspace U) of low-rank L, where L = U*Coeffs
%  
%   ----- USAGE -----
% INPUTS
%  Required:
%       M               = input data (observation matrix)
%       U               = estimated subspace Uhat from LoR_noisy()
%       k               = "block size" parameter for noisy-variant
%       noiselevel      = noise level sigma (std dev) used to generate noise
%       r               = rank of target low-rank component
%  Optional:');
%       verbose         = =1 to display messages, =0 to not
%  
% OUTPUTS');
%       Coeffs          = Recovered coordinates of L in subspace U
%  
%   [ See Supp B, Alg1 in: "Random Consensus Robust PCA" (Pimental 2017)
%     (https://danielpimentel.github.io/papers/R2PCA.pdf) ]
%

if nargin==0, dispFullUsage; return; end
if nargin<5, error('Not enough input args (run "help Sp_noisy" for usage)'); end
if nargin<6, verbose=0; end
if verbose>0, disp('--On function Sp...-'); end

[d,N] = size(M);        % Dimensions of the problem
% r = size(U,2);          % rank of the low-rank component
Coeffs = zeros(r,N);    % Matrix to keep the coefficients

% ======== Look one column at a time ========
for j=1:N         
    resp = 0;   % Have we already found k uncorrupted entries in this column?
    minerr = 1e10;  bestcoeffs = zeros(r,1);  % for unsuccessful
    tic; 
    % ======== Start looking for uncorrupted entries ========
    while resp==0 && toc<1e+2/N
        
        % == Take k random entries, and check if they are corrupted ==
        oi = randsample(d,k);
        Uoi = U(oi,:);
        xoi = M(oi,j);
        Coeffs(:,j) = (Uoi'*Uoi)\Uoi'*xoi;
        xoiPerp = xoi-Uoi*Coeffs(:,j);    
        
        % == If the entries are uncorrupted, use them to obtain a
        % == coefficient and move on to the next column
        if norm(xoiPerp)/norm(xoi) < noiselevel, resp = 1; end   

        currerr = norm(xoiPerp)/norm(xoi);
        if currerr<minerr
            minerr = currerr;
            bestcoeffs = Coeffs(:,j); 
        end
    end
    % if unsuccessful, use the coeffs that produced minimal error
    if resp==0 
        if verbose>0, fprintf(' using best coeffs for j=%d\n',j); end
        Coeffs(:,j) = bestcoeffs; 
    end
end

end


function dispUsage
    disp('function Coeffs = Sp_noisy(M, U, k, noiselevel, r[, verbose])');
    disp(' ');
    disp('  Recover Coeffs in subspace U of low-rank L, where L = U*Coeffs');
    disp(' ');
    disp('  ----- USAGE -----');
    disp('INPUTS');
    disp(' Required:');
    disp('      M               = input data (observation matrix)');
    disp('      U               = estimated subspace Uhat from LoR_noisy()');
    disp('      k               = "block size" parameter for noisy-variant');
    disp('      noiselevel      = noise level sigma (std dev) used to generate noise');
    disp('      r               = rank of target low-rank component');
    disp(' Optional:');
    disp('      verbose         = =1 to display messages, =0 to not');
    disp(' ');
    disp('OUTPUTS');
    disp('      Coeffs          = Recovered coordinates of L in subspace U');
    disp(' ');
    disp('  [ See Supp B, Alg1 in: "Random Consensus Robust PCA" (Pimental 2017)');
    disp('     (https://danielpimentel.github.io/papers/R2PCA.pdf) ]');
    disp(' ');
end

function dispFullUsage
    dispUsage;
    disp(' ');
    disp(' ============ 2nd part of R2PCA: recover coefficients ============');
    disp(' ====== Try to find r+1 uncorrupted entries in each column =======');
    disp(' ====== This entries determine the coefficient of the column =====');
    disp(' ');
    disp('Edited by:');
    disp('   rj 04-11-23');
    disp('     - modified noise-free version to account for noise');
    disp(' ');
end



