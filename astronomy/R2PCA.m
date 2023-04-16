function [L,S] = R2PCA(M,r)

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
%   M = matrix with a combination of low-rank plus sparse component
%   r = dimension of low-rank component.
%
% Output:
%
%   L = Low-rank component of M
%   S = Sparse component of M
%
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% =====================================================================

tic
Uhat = LoR(M,r);        % 1st part of R2PCA: recover basis of low-rank component
toc
tic
Coeffs = Sp(M,Uhat);    % 2nd part of R2PCA: recover coefficients
toc
% load('test_Coeffs.mat','Coeffs');
L = Uhat*Coeffs;        % Recover low-rank compoment
S = M-L;                % Recover sparse component
end


% ===== 1st part of R2PCA: recover basis of low-rank component =====
% ===== Try to find d-r uncorrupted (r+1)x(r+1) blocks (that are ===
% ===== independent). Each block will give us a projection of the ==
% ===== subspace. Then we "stitch" together all projections to =====
% ===== recover the subspace =======================================
function Uhat = LoR(M,r)

[d,N] = size(M);    % Dimensions of the problem
A = [];             % Matrix to store the information of the projections

%In this experiment, since things are so coherent, we might run into
%numerical issues if we just check rank(A).  So we can use an auxiliary
%matrix to verify whether Omega satisfies the subspace identifiability
%conditions in Lemma 1.
auxU = randn(d,r);
auxA = [];
itr=1;

auxrank = rank(auxA);

atic=tic;
% ======== Start looking for uncorrupted blocks ========
while auxrank<d-r
    
    % = Select a random (r+1)x(r+1) block and check if it is corrupted =
    oi = randsample(d,r+1);
    oj = randsample(N,r+1);
    
    % == If the block is uncorrupted, keep it to obtain a projection ==
    if rank(M(oi,oj))<=r
%         toc

        aoi = null(M(oi,oj)');
        aoi = aoi(:,1);
        A(oi,end+1) = aoi;
        
        auxaoi = null(auxU(oi,:)');
        auxA(oi,end+1) = auxaoi;

        auxrank = rank(auxA);
        
%         fprintf(' itr %d  auxrank %d\n',itr,auxrank);
        if mod(itr,10)==0, disp(itr); end
        itr=itr+1;
%         tic
    end
    
end
toc(atic)
% Stitch all the projections into the whole subspace, given by
% ker(A).  To avoid numerical issues, use svd instead of null.
[Uhat,~,~] = svd(A);
Uhat = Uhat(:,d-r+1:d);
end



% ============ 2nd part of R2PCA: recover coefficients ============
% ====== Try to find r+1 uncorrupted entries in each column =======
% ====== This entries determine the coefficient of the column =====
function Coeffs = Sp(M,U)


[d,N] = size(M);        % Dimensions of the problem
r = size(U,2);          % rank of the low-rank component
Coeffs = zeros(r,N);    % Matrix to keep the coefficients

% all_resp_vals = cell(1,N);


jtic=tic;
tic
% ======== Look one column at a time ========
for j=1:N

%     if mod(j,100)==0
%         jtoc=toc(jtic);
%         fprintf('%d, %g s\n',j,jtoc);
%         jtic=tic;
%     end
    disp(j);
    if mod(j,10)==0, toc; tic; end


%     fprintf(' j %d \n',j);
%     tic
    
    resp = 0;   % Have we already found r+1 uncorrupted entries in this column?
%     resp_vals = [];
    respcnt = 0;
    minerr = 1e6;
    errs = [];

    % ======== Start looking for uncorrupted entries ========
    while resp==0 % && length(resp_vals)<1e4
        
        % == Take r+1 random entries, and check if they are corrupted ==
        oi = randsample(d,r+1);
        Uoi = U(oi,:);
        xoi = M(oi,j);
        Coeffs(:,j) = (Uoi'*Uoi)\Uoi'*xoi;
        xoiPerp = xoi-Uoi*Coeffs(:,j);

%         resp_vals(end+1) = norm(xoiPerp)/norm(xoi);
        
        % == If the entries are uncorrupted, use them to obtain a
        % == coefficient and move on to the next column
        if norm(xoiPerp)/norm(xoi) < 1e-6 %1e-9
            resp = 1;
        end

        respcnt=respcnt+1;
        errs(end+1)=norm(xoiPerp)/norm(xoi);
        if norm(xoiPerp)/norm(xoi)<minerr
            minerr = norm(xoiPerp)/norm(xoi);
        end
%         if mod(respcnt,1e4)==0
%             disp(minerr);
%         end
        
    end
%     all_resp_vals{j} = resp_vals;
%     if mod(j,10)==0, toc; end
end
toc(jtic)

end





