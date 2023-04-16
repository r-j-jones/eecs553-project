% ====================================================================
% Sample code to replicate the astronomy experiments in
%
%   D. Pimentel-Alarcon, R. Nowak
%   Random Consensus Robust PCA,
%   International Conference on Artificial Intelligence and Statistics
%   (AISTATS), 2017.
%
% This code generates a video with blinking stars (background) and 
% moving objects (foreground), and then runs R2PCA to decompose this
% video (see Figures 5 and 6).
%
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% =====================================================================

clear all; close all; warning ('off','all'); clc;
rand('state',sum(100*clock));

addpath(genpath('/Users/robertjones/Desktop/W23/553/project/RPCA+MC_codes/inexact_alm_rpca'));

% ======================================================================
% =========================== GENERAL SETUP ============================
% ======================================================================

%%% LOAD REAL IMAGE DATA
testImageDir = '/Users/robertjones/Desktop/W23/553/project/data/wallflower/Bootstrap';
trueImagePath = [testImageDir filesep 'hand_segmented_00299.bmp'];
info = imfinfo(trueImagePath);

testImageNums = 150:349;
ntest = length(testImageNums);
dims = [info.Height, info.Width];
Data = zeros(dims(1)*dims(2), length(testImageNums),'single');
for t=1:ntest
    num = testImageNums(t);
    fname = [testImageDir filesep 'b' sprintf('%05d',num) '.bmp'];
    tmp = imread(fname);
%     tmp = imresize(tmp,0.5);
    tmp = im2double(tmp);
    tmp = rgb2gray(tmp);
    Data(:,t) = single(reshape(tmp,[],1));
end

M = Data;
if min(M(:))~=0 && max(M(:))~=1
    M = M - min(M);
    M = M / max(M(:));
end
% M = M';

%%% SET PARAMETERS FOR R2PCA
height = dims(1); 
width = dims(2);
frames = ntest;
r = 10;              % rank of the background

% ================== Run R2PCA and compute error ==================
tic
fprintf('Running R2PCA... \n');

lambda = -1;
tol = -1;
max_iter = -1;
rho = -1;
verbose = 1;


[A_hat, E_hat, iter] = inexact_alm_rpca_rj(double(M), lambda, tol, max_iter, rho, verbose);


[A_hatT, E_hatT, iterT] = inexact_alm_rpca_rj(double(M'), lambda, tol, max_iter, rho, verbose);



% [Lhat,Shat] = R2PCA(M,r);
% err = norm(L-Lhat,'fro')/norm(L,'fro');
% fprintf('Error = %1.1d. \n',err);

toc


if 0 == 1
% ======================= Display Video =======================
for t=1:frames
    
    % Combination of low-rank background plus sparse foreground
    figure(1);
    subplot(1,3,1);
%     imagesc(reshape(M(t,:),height,width));
    imagesc(reshape(M(:,t),height,width));
    colormap(gray); clim([0 1]);
    title({'Blinking stars','(low-rank background)','and moving objects','(sparse foreground)'});
    set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
    
    % Low-rank background recovered by R2PCA
    figure(1);
    subplot(1,3,2);
%     imagesc(reshape(abs(Lhat(t,:)),height,width));
    imagesc(reshape(abs(A_hat(:,t)),height,width));
    colormap(gray); clim([0 1]);
    title({'Blinking stars','(low-rank background)','recovered by R2PCA'});
    set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
    
    % Sparse foreground recovered by R2PCA
    figure(1);
    subplot(1,3,3);
%     imagesc(reshape(abs(Shat(t,:)),height,width));
    imagesc(reshape(abs(E_hat(:,t)),height,width));
    colormap(gray); clim([0 1]);
    title({'Moving objects','(sparse foreground)','recovered by R2PCA'});
    set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
    drawnow;
    pause(.04);
end

end



disp(' FINISHED - WAITING ---');




























