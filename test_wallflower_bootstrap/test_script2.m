

addpath('/Users/robertjones/Desktop/W23/553/project/clean_04112023');

cd /Users/robertjones/Desktop/W23/553/project/clean_04112023/test_wallflower_bootstrap

testImageDir = '/Users/robertjones/Desktop/W23/553/project/data/wallflower/Bootstrap';
trueImagePath = [testImageDir filesep 'hand_segmented_00299.bmp'];
info = imfinfo(trueImagePath);

testImageNums = 0:299;
ntest = length(testImageNums);
dims = [info.Height, info.Width];
Data = zeros(dims(1)*dims(2), length(testImageNums),'single');
% meanimg = zeros(dims(1),dims(2));
% fullimg = zeros(dims(1),dims(2), 3, length(testImageNums));
for t=1:ntest
    num = testImageNums(t);
    fname = [testImageDir filesep 'b' sprintf('%05d',num) '.bmp'];
    tmp = imread(fname);
%     tmp = imresize(tmp,0.5);
    tmp = im2double(tmp);
%     fullimg(:,:,:,t) = tmp;
%     meanimg = meanimg + tmp;
    tmp = rgb2gray(tmp);
%     tmp = imresize(Data,0.5);
    Data(:,t) = single(reshape(tmp,[],1));
end
clear tmp

% Data2 = zeros(dims(1)*dims(2), length(testImageNums),'single');
% for t=1:ntest
%     num = testImageNums(t);
%     fname = [testImageDir filesep 'b' sprintf('%05d',num) '.bmp'];
%     tmp = imread(fname);
%     tmp = im2double(tmp);
%     tmp = rgb2gray(tmp);
%     Data2(:,t) = single(reshape(tmp,[],1));
% end
% 
% 
% figure; 
% imshow(reshape(Data2(:,100),[120 160]),[]);
% 
% figure; 
% imshow(reshape(Data(:,100),[60 80]),[]);


% meanimg = meanimg/ntest;
% 
% figure; 
% imshow(meanimg,'border','tight');
% figure; 
% subplot(131); imshow(meanimg(:,:,1),'border','tight');
% subplot(132); imshow(meanimg(:,:,2),'border','tight');
% subplot(133); imshow(meanimg(:,:,3),'border','tight');
% 
% fmean = mean(fullimg,4);
% fstd = std(fullimg,0,4);
% 
% figure; 
% imshow(fmean,'border','tight');
% figure;
% subplot(131); imshow(fmean(:,:,1),[],'border','tight'); 
% subplot(132); imshow(fmean(:,:,2),[],'border','tight');
% subplot(133); imshow(fmean(:,:,3),[],'border','tight');
% 
% figure; 
% imshow(fstd,'border','tight');
% figure;
% subplot(131); imshow(fstd(:,:,1),[],'border','tight'); 
% subplot(132); imshow(fstd(:,:,2),[],'border','tight');
% subplot(133); imshow(fstd(:,:,3),[],'border','tight');


r_est = 5;
k = 10; 
% k = r_est*2;
n_est = 1e-1;

[ Uhat ] = LoR_noisy_test(Data, r_est, k, n_est);

[ Coeffs ] = Sp_noisy(Mgray, Uhat, k, sqrt(n_est), r_est);

Lhat = Uhat*Coeffs;
Shat = Mgray - Lhat;

figure;
subplot(121); imshow(gt); title('ground truth');
subplot(122); imshow(Shat); title('est. S');





%% RPCA-ALM
% addpath(genpath('/Users/robertjones/Desktop/W23/553/project/project_R2PCA'));
% [A_hat, E_hat, iter] = inexact_alm_rpca_rj(Mgray, 0.1);


%% Compare vs ground truth
trueImagePath = [testImageDir filesep 'hand_segmented_00299.bmp'];
trueimage = imread(trueImagePath);
gt = im2double(gt);
gt = rgb2gray(trueimage);

