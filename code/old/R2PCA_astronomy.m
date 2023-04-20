function [L, S, U, T, info] = R2PCA_astronomy(M, r, tol, timelimit, verbose)

% ====================================================================
% Decomposes a matrix M into its low-rank and sparse components using
% the simplest version of the R2PCA algorithm introduced in
%
%   D. Pimentel-Alarcon, R. Nowak
%   Random Consensus Robust PCA,
%   International Conference on Artificial Intelligence and Statistics
%   (AISTATS), 2017.
%
% Input:
%   
%   M = matrix with a combination of low-rank plus sparse component [2d-array]
%   r = dimension of low-rank component.                    [int]
%   tol[opt.] = tolerance for Sp (dist b/w m and u*theta) (def. 1e-9)[float]
%   timelimit[opt.] = use runtime limit (def. false) ***SEE NOTE BELOW***
%   verbose[opt.] = >0 displays messages, <0 does not (def. true) [int]
%
% Output: [arrays]
%
%   L = Low-rank component of M
%   S = Sparse component of M
%   U = PCA subspace
%   T = PCA coefficients
%
% =====================================================================
% *** timelimit *** notes:
%       -Valid class types: bool, numeric, stuct
%           - if bool: 
%               if true, uses time limit, sets maxtime=100s 
%               if false, does not use time limit
%           - if numeric:
%               if >0, uses time limit, sets maxtime=timelimit
%               if <=0, does not use time limit
%           - if struct:
%               must contain fields .timelimit and .maxtime
%
% =====================================================================
% 
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% 
% Edited by: Robert Jones
% email: rjjones@umich.edu
% 04-18-2023
% =====================================================================

if nargin<5, verbose = 1; end
if nargin<4, timelimit = false; end
if nargin<3, tol = 1e-9; end
if nargin<2
    fprintf('!!! not enough input args- run "help <funcname>" for usage\n');
    return
end

%%% Create Timer struct based on value of [optional] input arg 'timelimit'
Timer = makeTimer(timelimit);

%%% Initialize info struct
info = [];
info.Timer = Timer;
info.tol = tol;

%%% Run LoR function (recover pca subspace U)
if verbose>0, fprintf(' -Running LoR...\n'); end
[U, info.U] = LoR(M, r, Timer, verbose);        % 1st part of R2PCA: recover basis of low-rank component

%%% Decide whether to proceed to Sp function (recover pca coeffs T[heta])
if ~isempty(U) || info.U.completionFlag

    % If LoR completes successfully, continue on
    if verbose>0, fprintf('--Running Sp...\n'); end
    [T, info.T] = Sp(M, U, tol, Timer, verbose);    % 2nd part of R2PCA: recover coefficients
    
    if verbose>0, fprintf('--Recovering L & S...\n'); end
    L = U*T;        % Recover low-rank compoment
    S = M-L;                % Recover sparse component
else
    % If cant finish LoR in runtime limit, stop+exit++return empty matrices
    if verbose>0
        fprintf('--Did not finish LoR function!\n');
    end
    T=[];
    L=[];
    S=[];
    info.T.completionFlag = false;
end

end


% ===== 1st part of R2PCA: recover basis of low-rank component =====
% ===== Try to find d-r uncorrupted (r+1)x(r+1) blocks (that are ===
% ===== independent). Each block will give us a projection of the ==
% ===== subspace. Then we "stitch" together all projections to =====
% ===== recover the subspace =======================================
function [Uhat, info] = LoR(M, r, Timer, verbose)

if nargin<4, verbose=1; end
if nargin<3, Timer=[]; end

stic = tic;
info.elapTime = [];

[d,N] = size(M);    % Dimensions of the problem
A = zeros(d,0);             % Matrix to store the information of the projections

if d>N
    warning(' size(M,1)>size(M,2), may be slow, consider transposing ');
end

%In this experiment, since things are so coherent, we might run into
%numerical issues if we just check rank(A).  So we can use an auxiliary
%matrix to verify whether Omega satisfies the subspace identifiability
%conditions in Lemma 1.
auxU = randn(d,r);
auxA = [];
info.rankaux = rank(auxA); %to use for while loop criteria

info.itr = 0;  %track # of iterations until completion/termination
info.completionFlag = true; % Did we finish successfully or stop early?

tic;
% ======== Start looking for uncorrupted blocks ========
while info.rankaux<d-r
    info.itr = info.itr+1;

    % = Select a random (r+1)x(r+1) block and check if it is corrupted =
    oi = randsample(d,r+1);
    oj = randsample(N,r+1);
    
    % == If the block is uncorrupted, keep it to obtain a projection ==
    if rank(M(oi,oj))<=r
        aoi = null(M(oi,oj)');
        A(oi,end+1) = aoi;
%         A(oi,end+1) = aoi(:,1);
        
        auxaoi = null(auxU(oi,:)');
        auxA(oi,end+1) = auxaoi;
        info.rankaux = rank(auxA);

        if verbose>0
            fprintf(' itr=%d, rank(auxA)=%d\n',info.itr,info.rankaux);
        end
        
    end

    if ~isempty(Timer) && Timer.timelimit && toc>Timer.maxtime 
        if verbose>0, fprintf('--LoR Runtime limit reached, exiting\n'); end
        Uhat = [];
        info.completionFlag = false;
        info.elapTime = toc(stic);
        return
    end
    
end

% if favorable dims, compute rank(A) [skip if A is very large]
if d<N || numel(a)<5e5, info.rankA = rank(A); end

% Stitch all the projections into the whole subspace, given by
% ker(A).  To avoid numerical issues, use svd instead of null.
[Uhat,~,~] = svd(A);
Uhat = Uhat(:,d-r+1:d);

info.elapTime = toc(stic);
end



% ============ 2nd part of R2PCA: recover coefficients ============
% ====== Try to find r+1 uncorrupted entries in each column =======
% ====== This entries determine the coefficient of the column =====
function [Coeffs, info] = Sp(M, U, thr, Timer, verbose )

if nargin<5, verbose=1; end
if nargin<4, Timer=[]; end
if nargin<3, thr=1e-9; end

stic = tic;
info.elapTime = [];


[d,N] = size(M);        % Dimensions of the problem
r = size(U,2);          % rank of the low-rank component
Coeffs = zeros(r,N);    % Matrix to keep the coefficients
info.completionFlag = true;

info.itr = 0;
% ======== Look one column at a time ========
for j=1:N
    if verbose>0, fprintf('\tj=%d\n',j); end

    resp = 0;   % Have we already found r+1 uncorrupted entries in this column?
    
    % ======== Start looking for uncorrupted entries ========
    while resp==0
        info.itr = info.itr + 1;
        
        % == Take r+1 random entries, and check if they are corrupted ==
        oi = randsample(d,r+1);
        Uoi = U(oi,:);
        xoi = M(oi,j);
        Coeffs(:,j) = (Uoi'*Uoi)\Uoi'*xoi;
        xoiPerp = xoi-Uoi*Coeffs(:,j);
        
        % == If the entries are uncorrupted, use them to obtain a
        % == coefficient and move on to the next column
        if norm(xoiPerp)/norm(xoi) < thr
            resp = 1;
        end

        if ~isempty(Timer) && Timer.timelimit && toc>Timer.maxtime 
            if verbose>0, fprintf('--Sp Runtime limit reached, exiting\n'); end
            info.completionFlag = false;
            info.elapTime = toc(stic);
            return
        end
        
    end
end
info.elapTime = toc(stic);

end


function [T] = makeTimer(t)
% default maxtime is 100s
deftime = 100;
switch class(t)
    case 'numeric'
        if t>0
            T.timelimit=true;
            T.maxtime=t;
        else
            T.timelimit=false;
            T.maxtime=[];
        end
    case 'logical'
        T.timelimit = t;
        T.maxtime = deftime;
    case 'struct'
        T = t;
        if ~isfield(T,'timelimit'), T.timelimit = false; end
        if ~isfield(T,'maxtime'), T.maxtime = deftime; end
    otherwise
        T.timelimit=false;
        T.maxtime=deftime;
end
end




