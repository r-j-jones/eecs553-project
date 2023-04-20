% ====================================================================
addpath(genpath('\\engin-labs.m.storage.umich.edu\rjjones\windat.v2\Desktop\553\project\code\R2PCA_ORIG_rj'));
% =====================================================================

clear;
close all; 
warning ('off','all'); 
% clc;
% rand('state',sum(100*clock));

% ======================================================================

%Want to crop images?
DoCrop = true;

%Want to use parallelization?
UsePar = false;

%Want to downsample images? (Set ==1 to skip downsampling)
resizeFactor = 1;

%%% LOAD REAL IMAGE DATA
%Path to data dir
testImageDir = '\\engin-labs.m.storage.umich.edu\rjjones\windat.v2\Desktop\553\project\data_micro';
% testImageDir = '/RadOnc-MRI1/Student_Folder/rjones/RPCA/frames';

%Example image to load and get dims from
trueImagePath = [testImageDir filesep 'image0001.png'];
filebasename = 'image';


%Get image dims
info = imfinfo(trueImagePath);
height = info.Height;
width=info.Width;

%Want to crop image and process just a small piece of it?
if DoCrop
    xroi = 1:100; height = length(xroi);
    yroi = 1:100; width = length(yroi);
end

%Image numbers/indices to load/process
% testImageNums = 0:294;
testImageNums = 1:95;
ntest = length(testImageNums);

%Now load the images, preproc, vectorize + concat to mtx  
dims = [height/resizeFactor, width/resizeFactor];
M = zeros(dims(1)*dims(2), length(testImageNums)); %,'single');
for t=1:ntest
    num = testImageNums(t);
    fname = [testImageDir filesep filebasename sprintf('%04d',num) '.png'];
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
%     tmp = rgb2gray(tmp); %convert RGB to grayscale
    tmp = im2double(tmp); %convert uint8 grayscale to double(64)
    M(:,t) = (reshape(tmp,[],1)); %vectorize and hstack to mtx
end
M = M.';

% mask = zeros(size(tmp));
% mask(2:end,2:end) = 1;
% maskv = reshape(mask,[],1);
% Mcrop = M;
% Mcrop(:,maskv==0)=[];

% Mm = M - mean(M(:));
% figure; imagesc(reshape(abs(Mm(10,:)),dims))

%%% SET PARAMETERS FOR R2PCA
height = dims(1); 
width = dims(2);
frames = ntest;

%Set tolerance for Sp function in R2PCA()
tol = 1e-9;
r=2;  % rank of the background
timelimit=false;
verbose=1;

% ================== Run R2PCA and compute error ==================

ppool = parpool(10);

tic
% [L, S, U, T, info] = R2PCA_astronomy(M, r, tol, timelimit, verbose);
tol = 1e-3;
r=3;  % rank of the background
[L, S, U, T, info] = R2PCA_astronomy_par(M, r, tol, timelimit, verbose);
toc

%%%Display one image (set intensity windows to match range of M image)
figure('position',[679 380 1127 350]);
subplot(131);
imshow(reshape(M(10,:),[height width]),[],'border','tight');
cc=clim;
title('Original image'); set(gca,'FontSize',15);
subplot(132);
imshow(reshape(L(10,:),[height width]),cc,'border','tight');
title(['L, r=' num2str(r)]); set(gca,'FontSize',15);
subplot(133);
imshow(reshape((S(10,:)),[height width]),[],'border','tight');
title('S'); set(gca,'FontSize',15);
drawnow;

%%%Display the full M, Lhat, Shat matrices in subplots
figure;
subplot(311);
imagesc((M)); colormap gray;
cc=clim;
title('M frames'); set(gca,'FontSize',15);
subplot(312);
imagesc((L)); clim(cc); colormap gray;
title('Lhat frames, R2PCA'); set(gca,'FontSize',15);
subplot(313);
imagesc(abs(S)); clim([0 1]); colormap gray;
title('Shat frames, R2PCA'); set(gca,'FontSize',15);


if 1 == 1
    disp(' Playing R2PCA movie frames...in 5 seconds...');
    pause(1);

    figure('position',[113 552 1367 314],'color','w','InvertHardcopy','off');
    drawnow;
    % ======================= Display Video =======================
    for t=1:frames
        
        % Combination of low-rank background plus sparse foreground
%         figure(1);
        subplot(1,3,1);
        imagesc(reshape(M(t,:),height,width));
        colormap(gray); clim([0 1]); cc=clim;
        title({'M','True Image'});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
%         drawnow;
    
        % Low-rank background recovered by R2PCA
%         figure(1);
        subplot(1,3,2);
    %     imagesc(reshape(abs(Lhat(t,:)),height,width));
        imagesc(reshape((L(t,:)),height,width));
        colormap(gray); clim(cc);
        title({['L (r=' num2str(r) ')'],'(low-rank background)','recovered by RPCA-ALM'});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
%         drawnow;
    
        % Sparse foreground recovered by R2PCA
%         figure(1);
        subplot(1,3,3);
    %     imagesc(reshape(abs(Shat(t,:)),height,width));
        imagesc(reshape((S(t,:)),height,width));
        colormap(gray); clim([-1 1]); %clim(cc);
        title({'S','(sparse foreground)','recovered by RPCA-ALM'});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
        drawnow;
        pause(.02);
    end

end

notes='using while rankA<d-r in LoR';
save('GOOD_RESULTS_AHH_micro_ds2_100x100-roi_9.mat','L','S','M','T','U','xroi','yroi','info','tol','r','notes');

Lhat = U(:,1:4)*T(1:4,:);
Shat = M-Lhat;


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












