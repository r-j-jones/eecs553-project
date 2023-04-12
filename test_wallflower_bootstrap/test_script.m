

addpath('/Users/robertjones/Desktop/W23/553/project/clean_04112023');

cd /Users/robertjones/Desktop/W23/553/project/clean_04112023/test_wallflower_bootstrap

testImageDir = pwd; %'/Users/robertjones/Desktop/W23/553/project/data/wallflower/Bootstrap';
testImageName = 'b00299.bmp';
testImagePath = [testImageDir filesep testImageName];
testimage = imread(testImagePath);

% figure; 
% imshow(testimage,'Border','tight');
% title([testImageName ' (' class(testimage) ')']);
% set(gcf,'position',[481 178 567 452]);

% convert image from uint8 to double (requires Image Proc Toolbox)
M = im2double(testimage);
Mgray = rgb2gray(M);
Mmod = Mgray - mean(Mgray);


[COEFF, SCORE, LATENT] = pca(Mmod);
pctvar = zeros(size(LATENT));
for ii=1:length(LATENT)
    pctvar(ii) = sum(LATENT(1:ii))/sum(LATENT);
end

recimage = SCORE(:,1:5)*COEFF(:,1:5)';
figure;
subplot(121); imshow(recimage,[]); title('pca 5');
subplot(122); imshow(Mmod,[]); title('orig');


trueImageDir = pwd; %'/Users/robertjones/Desktop/W23/553/project/data/wallflower/Bootstrap';
trueImageName = 'hand_segmented_00299.bmp';
trueImagePath = [trueImageDir filesep trueImageName];
trueimage = imread(trueImagePath);
truegray = rgb2gray(trueimage);
truegray = im2double(truegray);


r_est = 10;
k = 20; 
% k = r_est*2;
% n_est = 0.1;
[n_est, ~, ~] = NoiseLevel(Mgray);

[ Uhat ] = LoR_noisy_test(Mgray, r_est, k, sqrt(n_est));
[ Coeffs ] = Sp_noisy(Mgray, Uhat, k, sqrt(n_est), r_est);

Lhat = Uhat*Coeffs;
Shat = Mgray - Lhat;

figure;
subplot(121); imshow(truegray); title('ground truth');
subplot(122); imshow(Shat); title('est. S');



addpath(genpath('/Users/robertjones/Desktop/W23/553/project/project_R2PCA'));

[A_hat, E_hat, iter] = inexact_alm_rpca_rj(Mgray, 0.1);

