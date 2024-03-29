
% addpath /Users/robertjones/Desktop/W23/553/project/updated_041723
addpath(genpath('/Users/robertjones/Desktop/W23/553/project/R2PCA_ORIG_rj'));

vidDir = '/Users/robertjones/Desktop/W23/553/project/data/bmc';
vidPath = [vidDir filesep '111.mp4'];
vidObj = VideoReader(vidPath);

frames = read(vidObj,[250 499]);
dims = size(frames,1:2);
height = dims(1); 
width = dims(2);
nframes = size(frames,4);

resizeFactor = 2;
h = height/resizeFactor;
w = width/resizeFactor;
images = zeros(h,w,nframes);

for f=1:nframes
    tmp = frames(:,:,:,f);
    if resizeFactor>1
        tmp = imresize(tmp,1/resizeFactor);
    end
    tmp = rgb2gray(tmp);
    tmp = im2double(tmp);
    images(:,:,f) = im2double(tmp);
end

images = reshape(images,[],nframes);
images = images.';



r=5;
tol=1e-9;
timelimit=false;
verbose=0;

% [L,S,Coeffs,Uhat] = R2PCA(images',r);
[L, S, U, T, info] = R2PCA_astronomy(images, r, tol, timelimit, verbose);

S_ = abs(S);
S_(S_<tol)=0;
S_(S_>0)=1;

% figure; imshow(reshape(L(150,:),h,w),[]);
% figure; imshow(abs(reshape(S(150,:),h,w)),[]);
% figure; imshow(reshape(images(150,:),h,w),[]);

[ gt ] = load_validation_data('111');
gt = gt.';


if 1 == 1
    disp(' Playing R2PCA movie frames......');
    pause(1);

    figure('position',[518 151 1039 715],'color','w','InvertHardcopy','off'); %[113 552 1367 314]
    drawnow;
    % ======================= Display Video =======================
    for t=1:nframes
        
        % Combination of low-rank background plus sparse foreground
        figure(1);
        subplot(2,2,1);
        imagesc(reshape(images(t,:),h,w));
        colormap(gray); cc=clim;
        title({'M','True Image',''});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
%         drawnow;
    
        % Low-rank background recovered by R2PCA
        figure(1);
        subplot(2,2,3);
    %     imagesc(reshape(abs(Lhat(t,:)),height,width));
        imagesc(reshape((L(t,:)),h,w));
        colormap(gray); clim(cc);
        title({['L (r=' num2str(r) ')'],'(low-rank background)','recovered by R2PCA'});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
%         drawnow;
    
        % Sparse foreground recovered by R2PCA
        figure(1);
        subplot(2,2,2);
    %     imagesc(reshape(abs(Shat(t,:)),height,width));
        imagesc(reshape(abs(S(t,:)),h,w));
        colormap(gray); clim(cc);
        title({'S','(sparse foreground)','recovered by R2PCA'});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
%         drawnow;

        %Ground truth sparse outliers
        figure(1);
        subplot(2,2,4);
    %     imagesc(reshape(abs(Shat(t,:)),height,width));
        imagesc(reshape(abs(gt(t,:)),h,w));
        colormap(gray); clim(cc);
        title({'S_{true}','(sparse ground truth)',''});
        set(gca, 'XTickLabelMode','manual','XTickLabel',[],'YTickLabelMode','manual','YTickLabel',[]);
        drawnow;
        pause(.02);
    end

end



%%%Display the full M, Lhat, Shat matrices in subplots
figure;
subplot(311);
imagesc((images')); colormap gray;
cc=clim;
title('M frames'); set(gca,'FontSize',15);
subplot(312);
imagesc((L)); clim(cc); colormap gray;
title('Lhat frames, R2PCA'); set(gca,'FontSize',15);
subplot(313);
imagesc((S)); %clim(cc); 
colormap gray;
title('Shat frames, R2PCA'); set(gca,'FontSize',15);




%% Compare to ground truth

[ gt ] = load_validation_data('111');

gt_bin = double(gt>0);
gt_bin = gt_bin';

S_bin = double(abs(S)>0);


S_err = norm(gt_bin-S_bin,"fro")/norm(gt_bin,"fro");


figure; 
subplot(311);
imagesc(gt_bin); title('gt');
subplot(312);
imagesc(S_bin); title('S');
subplot(313);
imshowpair(gt_bin,S_bin,'diff'); title('diff');





