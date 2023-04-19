function [L,S,U,T,info] = R2PCA(M, r, timelimit, verbose)

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
%   r = dimension of low-rank component. [int]
%   UseTimiLimit (opt.) = set 100 second runtime limit for LoR & Sp functions
%                            [bool]
%   verbose [opt.] = >0 displays messages, <0 does not [int]
%
% Output: [arrays]
%
%   L = Low-rank component of M
%   S = Sparse component of M
%   U = PCA subspace
%   T = PCA coefficients
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

if nargin<4, verbose = 1; end
if nargin<3, timelimit = true; end
if nargin<2
    fprintf('!!! not enough input args - run help R2PCA for usage\n');
    return
end

%%% Create Timer struct based on value of [optional] input arg 'timelimit'
Timer = makeTimer(timelimit);

%%% Initialize info struct
info = [];
info.Timer = Timer;

if verbose>0, fprintf(' -Running LoR...\n'); end
[U, info.U] = LoR(M, r, Timer, verbose);        % 1st part of R2PCA: recover basis of low-rank component

if ~isempty(U)
    % If LoR completes successfully, continue on
    if verbose>0, fprintf('--Running Sp...\n'); end
    [T, info.T] = Sp(M, U, Timer, verbose);    % 2nd part of R2PCA: recover coefficients
    
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
    info.T = [];
end

end


% ===== 1st part of R2PCA: recover basis of low-rank component =====
% ===== Try to find d-r uncorrupted (r+1)x(r+1) blocks (that are ===
% ===== independent). Each block will give us a projection of the ==
% ===== subspace. Then we "stitch" together all projections to =====
% ===== recover the subspace =======================================
function [Uhat, info] = LoR(M,r,Timer,verbose)

if nargin<4, verbose=1; end
if nargin<3, Timer.timelimit=true; end

stic = tic;
info.elapTime = [];

[d,N] = size(M);    % Dimensions of the problem
A = zeros(d,0);             % Matrix to store the information of the projections
rankA = rank(A);
info.completionFlag = true; % Did we finish successfully or stop early?

info.itr = 0;
tic;
% ======== Start looking for uncorrupted blocks ========
while rankA<d-r
    info.itr = info.itr + 1;

    % = Select a random (r+1)x(r+1) block and check if it is corrupted =
    oi = randsample(d,r+1);
    oj = randsample(N,r+1);
    
    % == If the block is uncorrupted, keep it to obtain a projection ==
    if rank(M(oi,oj))==r
        aoi = null(M(oi,oj)');
        A(oi,end+1) = aoi;
        if verbose>0, fprintf(' -rank(A)=%d\n',rankA); end
    end

    if Timer.timelimit && toc>100
        if verbose>0, fprintf('--Runtime limit reached, exiting\n'); end
        A = [];
        info.completionFlag = false;
        info.elapTime = toc(stic);
        return
    end
    
end

% Stitch all the projections into the whole subspace, given by
% ker(A).
Uhat = null(A');
info.elapTime = toc(stic);

end



% ============ 2nd part of R2PCA: recover coefficients ============
% ====== Try to find r+1 uncorrupted entries in each column =======
% ====== This entries determine the coefficient of the column =====
function [Coeffs, info] = Sp(M, U, Timer, verbose, thr)

if nargin<5, thr=1e-9; end
if nargin<4, verbose=1; end
if nargin<3, Timer.timelimit=true; end

stic = tic;
info.elapTime = [];

[d,N] = size(M);        % Dimensions of the problem
r = size(U,2);          % rank of the low-rank component
Coeffs = zeros(r,N);    % Matrix to keep the coefficients
info.completionFlag = true; % Did we finish successfully or stop early?

info.itr = 0;
% ======== Look one column at a time ========
for j=1:N
    if verbose>0, fprintf('\tj=%d\n',j); end
    resp = 0;   % Have we already found r+1 uncorrupted entries in this column?
    
    tic;
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

        if Timer.timelimit && toc>1e+2/N
            info.completionFlag = false;
            break
        end
        
    end
end
info.elapTime = toc(stic);

end


function [T] = makeTimer(t)
% default maxtime is 100s
deftime = 100;
switch class(t)
    case numeric
        if t>0
            T.timelimit=true;
            T.maxtime=t;
        else
            T.timelimit=false;
            T.maxtime=[];
        end
    case logical
        T.timelimit = t;
        T.maxtime = deftime;
    case struct
        T = t;
        if ~isfield(T,'timelimit'), T.timelimit = false; end
        if ~isfield(T,'maxtime'), T.maxtime = deftime; end
    otherwise
        T.timelimit=false;
        T.maxtime=deftime;
end
end


