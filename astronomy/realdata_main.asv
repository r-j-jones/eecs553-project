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

% clear all; 
close all; 
warning ('off','all'); 
% clc;
rand('state',sum(100*clock));

% ======================================================================
% =========================== GENERAL SETUP ============================
% ======================================================================

%%% LOAD REAL IMAGE DATA
% testImageDir = '/Users/robertjones/Desktop/W23/553/project/data/wallflower/Bootstrap';
testImageDir = '/RadOnc-MRI1/Student_Folder/rjones/RPCA/Camouflage';

% trueImagePath = [testImageDir filesep 'hand_s/egmented_00299.bmp'];
trueImagePath = [testImageDir filesep 'hand_segmented_00251.BMP'];

info = imfinfo(trueImagePath);

% testImageNums = 0:294;
testImageNums = 241:280;

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
M = M';

%%% SET PARAMETERS FOR R2PCA
height = dims(1); 
width = dims(2);
frames = ntest;
r = 10;              % rank of the background

% ================== Run R2PCA and compute error ==================


if isempty(gcp)
    ppool = parpool(12);
end


fprintf('Running R2PCA... \n');
% [Lhat,Shat] = R2PCA(M,r);

tic

tol = 1e-6;
[Lhat,Shat] = R2PCA_par(M,r, tol);
% err = norm(L-Lhat,'fro')/norm(L,'fro');
% fprintf('Error = %1.1d. \n',err);
toc

% save('camoflauge_test_Sp_tol1e-6_041623_results.mat','Lhat','Shat','elaptime','Sp_elaptime','testImageNums','tol')


figure;
subplot(131);
imshow(reshape(M(10,:),[height width]),[0 1],'border','tight');
title('Original image'); set(gca,'FontSize',15);
subplot(132);
imshow(reshape(Lhat(10,:),[height width]),[0 1],'border','tight');
title('L'); set(gca,'FontSize',15);
subplot(133);
imshow(reshape(Shat(10,:),[height width]),[0 1],'border','tight');
title('S'); set(gca,'FontSize',15);

figure;
subplot(311);
imagesc((M)); caxis([0 1]); colormap gray;
title('M frames'); set(gca,'FontSize',15);
subplot(312);
imagesc((Lhat)); caxis([0 1]); colormap gray;
title('L frames'); set(gca,'FontSize',15);
subplot(313);
imagesc((Shat)); caxis([0 1]); colormap gray;
title('S frames'); set(gca,'FontSize',15);


if 0 == 1
res=load("test_coeffs_with_1e-6.mat",'Coeffs','U');
Lhat = Uhat*Coeffs;        % Recover low-rank compoment
Shat = M-Lhat;  
figure;
subplot(131);
imshow(reshape(M(10,:),[height width]),[0 1],'border','tight');
title('Original image'); set(gca,'FontSize',15);
subplot(132);
imshow(reshape(Lhat(10,:),[height width]),[0 1],'border','tight');
title('L'); set(gca,'FontSize',15);
subplot(133);
imshow(reshape(Shat(10,:),[height width]),[0 1],'border','tight');
title('S'); set(gca,'FontSize',15);
end

if 0 == 1
% ======================= Display Video =======================
for t=1:frames
    
    % Combination of low-rank background plus sparse foreground
    figure(1);
    subplot(1,3,1);
    imagesc(reshape(M(t,:),height,width));
    colormap(gray); clim([0 1]);
    title({'Blinking stars','(low-rank background)','and moving objects','(sparse foreground)'});
    set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
    
    % Low-rank background recovered by R2PCA
    figure(1);
    subplot(1,3,2);
    imagesc(reshape(abs(Lhat(t,:)),height,width));
    colormap(gray); clim([0 1]);
    title({'Blinking stars','(low-rank background)','recovered by R2PCA'});
    set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
    
    % Sparse foreground recovered by R2PCA
    figure(1);
    subplot(1,3,3);
    imagesc(reshape(abs(Shat(t,:)),height,width));
    colormap(gray); clim([0 1]);
    title({'Moving objects','(sparse foreground)','recovered by R2PCA'});
    set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
    
    drawnow;
    pause(.04);
end

end



disp(' FINISHED - WAITING ---');




























