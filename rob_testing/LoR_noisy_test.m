function [ Uhat ] = LoR_noisy_test(M, r, k, noiselevel, verbose)
%
% ---FOR TESTING/DEBUGGING PURPOSES----
% 
% function Uhat = LoR_noisy(M,r,k,noiselevel[,verbose])
%  
%   Reconstruct subspace U from noisy data matrix M using R2PCA
% 
%   ----- USAGE -----
% INPUTS
%  Required:
%       M               = input data (observation matrix)
%       r               = rank of target low-rank component
%       k               = "block size" parameter for noisy-variant
%       noiselevel      = noise level sigma (std dev) used to generate noise
%  Optional:
%       verbose         = =1 to display messages, =0 to not
%
% OUTPUTS
%       Uhat            = Reconstructed subspace U
% 
%   [ See Supp B, Alg1 in: "Random Consensus Robust PCA" (Pimental 2017)
%      (https://danielpimentel.github.io/papers/R2PCA.pdf) ]

if nargin==0, dispFullUsage; return; end
if nargin<4, error('Not enough input args (run "help LoR_noisy" for usage)'); end
if nargin<5, verbose=0; end
if verbose>0, fprintf('\n---In LoR function...---\n'); end

[d,N] = size(M);    % Dimensions of the problem
A = zeros(d,0,'single');     % Matrix to store the information of the projections

otic = tic;

while rank(A)<d-r
    % Loop to get d-r subspace projs (used to "stitch" together U)
    for outerind=1:d-r
    %     disp(outerind); 
        itic = tic;
    
        % * omega_i : r+1 random column indices
        omega_i = sort(randsample(d,r+1));
        
        % * kappa_i : k random indices, including omega_i
        new_inds = randsample(setdiff(1:d, omega_i), k-r-1);
        kappa_i = sort(union(omega_i, new_inds));
        if length(kappa_i)~=length(unique(kappa_i)) || length(kappa_i)~=k
            warning('Error- kappa_i has incompatible dimension (should be length k)');
        end
        
        % * M_kappa_i : extract kappa_i rows from data matrix M 
        M_kappa_i = M(kappa_i,:);
    
        %%% Repeat: extract k random cols of M_kappa_i <- M'_kappa_i
        %%% Until:  (r+1)th sing val of M'_kappa_i <= noise level
        %    
        sval_rplus1 = noiselevel * 1e3;  % note: arbitrary starting value 
    %     cnt=0; errs=[];
        while sval_rplus1>noiselevel
            kcols = sort(randsample(N,k));       % this is random "k" col inds in paper
            M_prime_kappa_i = M_kappa_i(:,kcols);      % M'_kappa_i matrix
    %         svals = svds(M_prime_kappa_i, r+1);   % the first r+1 svdvals of M'_kappa_i
            [~,svals,~] = svd(M_prime_kappa_i,"econ","vector");
            sval_rplus1 = svals(end);    % the r+1th svdval 
    %         cnt=cnt+1; errs(end+1)=sval_rplus1;
    %         if mod(cnt,1000)==0
    %             fprintf('%g\n',sval_rplus1);
    %         end
        end
    %     disp('svalr1 found');
    
        % Get r-leading left singular vectors of M'_kappa_i [Alg1 lines 14-15]
        [Vtmp,~,~] = svd(M_prime_kappa_i);
        V_kappa_i = Vtmp(:,1:r);
    
        % * v_i : extract subset of r entries of omega_i [Alg1 lines 16-17]
        %  [NOTE: Alg1 says "subset of kappa_i", but it SHOULD BE omega_i]
        v_i = sort(randsample(omega_i, r));
    
        % vec of j values : find elements of ki not in vi [Alg1 line 18]
        js = setdiff(kappa_i, v_i);
        
        % iterate through j (element of kappa_i NOT IN v_i) [Alg1 line 18]
        for ind=1:length(js)
            % an 
            j = js(ind);
            % omega_ij = (v_i UNION j)  [Alg1 line 19]
            omega_ij = sort(union(v_i, j));
            % V_omega_ij = omega_ij rows of V_kappa_i
            V_omega_ij = V_kappa_i(ismember(kappa_i,omega_ij),:);
            % nonzero vec in ker(transpose(V_omega_ij)) [Alg1 lines 20-21]
            a_omega_ij = null(V_omega_ij');
            % double check a_omega_ij is nonzero vector
            if ~isempty(a_omega_ij) && (nnz(a_omega_ij==0)~=numel(a_omega_ij))
                % insert a_omega_ij into A [Alg1 lines 22-23]
                aij = zeros(d,1);
                aij(omega_ij) = a_omega_ij;
                A(:,end+1) = single(aij);
            end
        end
        itoc = toc(itic); if mod(outerind,1000)==0,disp(itoc); end
    end
end

otoc = toc(otic);
disp(otoc);

% Get subspace Uhat = last r left SingVecs of A (which appx.'s ker(A.T)) 
[Ua, ~, ~] = svd(A);
[Ua, ~, ~] = svd((A),"econ","vector");
Uhat = Ua(:,end-r+1:end);
% Uhat = orth(Uhat);

end



function dispUsage
    disp('function Uhat = LoR_noisy(M,r,k,noiselevel[,verbose])');
    disp(' ');
    disp('  Reconstruct subspace U from noisy data matrix M using R2PCA');
    disp(' ');
    disp('  ----- USAGE -----');
    disp('INPUTS');
    disp(' Required:');
    disp('      M               = input data (observation matrix)');
    disp('      r               = rank of target low-rank component');
    disp('      k               = "block size" parameter for noisy-variant');
    disp('      noiselevel      = noise level sigma (std dev) used to generate noise');
    disp(' Optional:');
    disp('      verbose         = =1 to display messages, =0 to not');
    disp(' ');
    disp('OUTPUTS');
    disp('      Uhat            = Reconstructed subspace U');
    disp(' ');
    disp('  [ See Supp B, Alg1 in: "Random Consensus Robust PCA" (Pimental 2017)');
    disp('     (https://danielpimentel.github.io/papers/R2PCA.pdf) ]');
    disp(' ');
end

function dispFullUsage
    dispUsage;
    disp(' ');
    disp('===== 1st part of R2PCA: recover basis of low-rank component =====');
    disp('===== Try to find d-r uncorrupted (r+1)x(r+1) blocks (that are ===');
    disp('===== independent). Each block will give us a projection of the ==');
    disp('===== subspace. Then we "stitch" together all projections to =====');
    disp('===== recover the subspace =======================================');
    disp(' ');
    disp('Edited by:');
    disp('   rj 04-11-23');
    disp('     - modified noise-free version to account for noise');
    disp(' ');
end




