% ====================================================================
% Sample code to replicate the astronomy experiments in
% % % EDIT: rj, 4/17/23
% % % %    - load microscopy data and perform r2pca.
% % % %    -- NOTE: i tested the R2PCA_par() vs R2PCA() functions, and they
%           perform differently, suggesting somethings up with the parallel
%           implementation. R2PCA_par returned erratic high error values on
%           some trials, while others it returned very similar values as
%           R2PCA (no parallelization) [which were around 1e-13, very low].
%          - I used the R2PCA_par function to process the microscopy data.
%            Using r=1 (low rank = 1), the results were very good in some
%            frames and terrible in others. This again hints that something
%            is wrong with the par version. 
%          - I preprocessed microscopy images:
%          ---FOR EACH IMAGE,
% %             - downsampling each image by factor of 2
%                   [imresize(img,0.5)], 
%
%               - rgb2gray()
%               - im2double()
%               - vectorized and added to matrix M
%         ---ONCE M matrix is formed:
%               - Subtract off mean of entire matrix (so that mean(M(:))~0)
%               - ?? Divide by max value of M (so that max(M(:))=1)
%               - transpose M (so that dims are 299 x TotalNbPixels
%         -Then, I set r=1, tol=1e-9 (tol is for the Sp function for
%               computing coefficients)
%   --- NOTE ---: When i kept the data as class==double, r2pca worked with
%                   tol=1e-9. When I made class(M)==single, it did not work
%                   and gave bad results. I ran this on our research groups
%                   computer with 100s GB memory so was no problem, but
%                   might be limiting on laptop...
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

clear;
close all; 
warning ('off','all'); 
% clc;
rand('state',sum(100*clock));

% ======================================================================


%%% LOAD REAL IMAGE DATA
%Path to data dir
% testImageDir = '/Users/robertjones/Desktop/W23/553/project/data/wallflower/Bootstrap';
testImageDir = '/RadOnc-MRI1/Student_Folder/rjones/RPCA/frames';

%Example image to load and get dims from
% trueImagePath = [testImageDir filesep 'hand_s/egmented_00299.bmp'];
trueImagePath = [testImageDir filesep 'Pond Water Brightfield 20x_frame_0.png'];
filebasename = 'Pond Water Brightfield 20x_frame_';

%Factor to downsample each image by
resizeFactor = 2;

%Get image dims
info = imfinfo(trueImagePath);
height = info.Height;
if resizeFactor==10
    width = 960;
else
    width=info.Width;
end

%Want to crop image and process just a small piece of it?
DoCrop = false;
if DoCrop
    xroi = 2:161; height = length(xroi);
    yroi = 551:720; width = length(yroi);
end

%Image numbers/indices to load/process
% testImageNums = 0:294;
testImageNums = 0:298;
ntest = length(testImageNums);

%Now load the images, preproc, vectorize + concat to mtx  
dims = [height/resizeFactor, width/resizeFactor];
M = zeros(dims(1)*dims(2), length(testImageNums)); %,'single');
for t=1:ntest
    num = testImageNums(t);
    fname = [testImageDir filesep filebasename sprintf('%d',num) '.png'];
    tmp = imread(fname); %load the image
    if resizeFactor==10 %special case for extreme downsampling, to make dims multiple of 10
        tmp = tmp(:,1:960,:);
    end
    if DoCrop  %If cropping, do it now
        tmp = tmp(xroi,yroi,:);
    end
    if resizeFactor>1 %If resizing, do it now (can play with 'bilinear' vs 'nearest' interpolation
        tmp = imresize(tmp,1/resizeFactor,'bilinear');
    end
    tmp = rgb2gray(tmp); %convert RGB to grayscale
    tmp = im2double(tmp); %convert uint8 grayscale to double(64)
    M(:,t) = (reshape(tmp,[],1)); %vectorize and hstack to mtx
end

% Display first image
figure; imshow(reshape(M(:,1),dims),[]);

%Subtract off mean, normalize so max intensity ==1
M = M-mean(M(:));
M = M/max(M(:));
%Display first image after final preproc
figure; imshow(reshape(M(:,1),dims),[]);

%Alternate intensity normalization
% if min(M(:))~=0 && max(M(:))~=1
% %     M = M - min(M);
%     M = M / max(M(:));
% end

%Transpose matrix (as done in astronomy demo, makes LoR faster, but Sp slower) 
M = M';

%%% SET PARAMETERS FOR R2PCA
height = dims(1); 
width = dims(2);
frames = ntest;

%Set tolerance for Sp function in R2PCA()
tol = 1e-9;

%Run R2PCA with different values of r
for r=1 %[1,3,4]
% r = 50;              % rank of the background

% ================== Run R2PCA and compute error ==================

%%% IF RUNNING PARALLEL VERSION, CREATE PARPOOL IF IT DNE %%%
if isempty(gcp)
    ppool = parpool(12);
end

fprintf('Running R2PCA... \n');
% [Lhat,Shat] = R2PCA(M,r);
tic

%%%%RUN R2PCA parallel version
% [Lhat,Shat] = R2PCA_par(double(M),r, tol);

%%%%RUN R2PCA regular (non-parallel) version
[Lhat,Shat] = R2PCA(double(M),r);

% %Display error (N/A HERE - NO GROUND TRUTH L LIKE WE HAD IN SIMULATIONS)
% err = norm(L-Lhat,'fro')/norm(L,'fro');
% fprintf('Error = %1.1d. \n',err);

toc

%%%Display one image (set intensity windows to match range of M image)
figure('position',[679 380 1127 350]);
subplot(131);
imshow(reshape(M(10,:),[height width]),[],'border','tight');
cc=clim;
title('Original image'); set(gca,'FontSize',15);
subplot(132);
imshow(reshape(Lhat(10,:),[height width]),cc,'border','tight');
title(['L, r=' num2str(r)]); set(gca,'FontSize',15);
subplot(133);
imshow(reshape(Shat(10,:),[height width]),cc,'border','tight');
title('S'); set(gca,'FontSize',15);
drawnow;

%%%Display the full M, Lhat, Shat matrices in subplots
figure;
subplot(311);
imagesc((M)); colormap gray;
cc=clim;
title('M frames'); set(gca,'FontSize',15);
subplot(312);
imagesc((Lhat)); clim(cc); colormap gray;
title('Lhat frames, R2PCA'); set(gca,'FontSize',15);
subplot(313);
imagesc((Shat)); clim(cc); colormap gray;
title('Shat frames, R2PCA'); set(gca,'FontSize',15);


if 1 == 1
    disp(' Playing R2PCA movie frames...in 5 seconds...');
    pause(5);

    figure('position',[113 552 1367 314],'color','w','InvertHardcopy','off');
    % ======================= Display Video =======================
    for t=1:frames
        
        % Combination of low-rank background plus sparse foreground
        figure(1);
        subplot(1,3,1);
        imagesc(reshape(M(t,:),height,width));
        colormap(gray); cc=clim;
        title({'M','True Image'});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
        drawnow;
    
        % Low-rank background recovered by R2PCA
        figure(1);
        subplot(1,3,2);
    %     imagesc(reshape(abs(Lhat(t,:)),height,width));
        imagesc(reshape((Lhat(t,:)),height,width));
        colormap(gray); clim(cc);
        title({['L (r=' num2str(r) ')'],'(low-rank background)','recovered by RPCA-ALM'});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
        drawnow;
    
        % Sparse foreground recovered by R2PCA
        figure(1);
        subplot(1,3,3);
    %     imagesc(reshape(abs(Shat(t,:)),height,width));
        imagesc(reshape((Shat(t,:)),height,width));
        colormap(gray); clim(cc);
        title({'S','(sparse foreground)','recovered by RPCA-ALM'});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
        drawnow;
        pause(.04);
    end

end

end

% save('better_res_r=1_tol=1e-9.mat','Lhat','Shat','tol','r','testImageNums','resizeFactor','xroi','yroi');
% 
% save('good_res_r=3_tol=1e-9.mat','Lhat','Shat','tol','r');
% save('decent_res_r=2_tol=1e-9.mat','Lhat','Shat','tol','r');


% %% SET PARAMETERS FOR RPCA-ALM
% lambda = -1;
% tol = 1e-9;
% max_iter = -1;
% rho = -1;
% verbose = 1;
% 
% %% ================== Run RPCA-ALM and compute error ==================
% tic
% 
% addpath(genpath('/RadOnc-MRI1/Student_Folder/rjones/RPCA/inexact_alm_rpca'));
% 
% fprintf('Running RPCA-ALM... \n');
% [A_hat, E_hat, niters] = inexact_alm_rpca_rj(double(M), lambda, tol, max_iter, rho, verbose);
% 
% toc
% 
% % save('camoflauge_test_Sp_tol1e-6_041623_results.mat','Lhat','Shat','elaptime','Sp_elaptime','testImageNums','tol')
% 
% figure;
% subplot(131);
% imshow(reshape(M(10,:),[height width]),[0 1],'border','tight');
% title('Original image'); set(gca,'FontSize',15);
% subplot(132);
% imshow(reshape(A_hat(10,:),[height width]),[0 1],'border','tight');
% title('L'); set(gca,'FontSize',15);
% subplot(133);
% imshow(reshape(E_hat(10,:),[height width]),[0 1],'border','tight');
% title('S'); set(gca,'FontSize',15);









disp(' FINISHED - WAITING ---');









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




















