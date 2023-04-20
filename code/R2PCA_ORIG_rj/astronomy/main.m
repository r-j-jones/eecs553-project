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


%% Set parameters for simulations

% ======================================================================
% =========================== GENERAL SETUP ============================
% ======================================================================
height = 90;        % height of each video frame
width = 120;        % width of each video frame
frames = 100;       % number of frames in the video

r = 5;              % rank of the background
numStars = 500;     % Number of blinking "stars" in the background. In the paper we play with a range between 10 and 10800. This determines how coherent is the background. The smaller NumStars, the more coherent.
maxSpeed = 5;       % How fast each object will be moving 
numObjects = 100;   % Number of objects that will be moving around. In the paper we play with a range between 1 and 400.
objectWidth = 5;    % Width of each object (in number of pixels)


r = 10;
numStars = 10000;
maxSpeed = 10;
numObjects = 100;
objectWidth = 10;    


%% Generate simulated data matrices

% ======================================================================
% ============================= EXPERIMENT =============================
% ======================================================================

% ========== Low-rank matrix (background with blinking stars) ==========
fprintf('Creating low-rank matrix with blinking stars... \n');
dim = height*width;       % Ambient dimension
U = abs(randn(dim,r));    % Basis of r-dimensional subspace
stars = randsample(dim,min(numStars,dim));  % Location of stars
U(stars,:) = 100*abs(randn(min(numStars,dim),r)); %basis has more energy on rows corresponding to "stars" pixels

Theta = abs(randn(r,frames));   % Coefficients of each frame with respect to the low-rank basis
L = U*Theta;                    % Low-rank matrix

%At this point, each frame is a column and each row is a pixel. We transpose everything to speed things up.
L = L';
N = height*width;
dim = frames;
L = L/max(max(L));  %normalize for imaging purposes.

% =============== Sparse matrix (moving objects) ===============
fprintf('Creating sparse matrix with moving objects... \n');
S = zeros(dim,N);
for no = 1:numObjects,
    %==Create Starting Point and Direction of Moving Object==
    %Will it start from top, bottom, left or right?
    tblr = randi(4);
    switch tblr
        case 1,
            object.x0 = randi(width);
            object.y0 = height;
            object.deltax = sign(randn)*randi(maxSpeed);
            object.deltay = -randi(maxSpeed);
        case 2,
            object.x0 = randi(width);
            object.y0 = 1;
            object.deltax = sign(randn)*randi(maxSpeed);
            object.deltay = randi(maxSpeed);
        case 3,
            object.x0 = 1;
            object.y0 = randi(height);
            object.deltax = randi(maxSpeed);
            object.deltay = sign(randn)*randi(maxSpeed);
        case 4,
            object.x0 = width;
            object.y0 = randi(height);
            object.deltax = -randi(maxSpeed);
            object.deltay = sign(randn)*randi(maxSpeed);
    end
    
    %Put object in frames, and vectorize frames to form S
    object.startingTime = randi(dim);
    for t=1:dim,
        object.x = object.x0 + object.deltax*(t-object.startingTime);
        object.y = object.y0 + object.deltay*(t-object.startingTime);
        
        foreground = zeros(height,width);
        
        if object.x>=1 && object.x<=width && object.y>=1 && object.y<=height,
            foreground(round(object.y),round(object.x)) = 1;
            foreground = conv2(foreground,ones(objectWidth),'same');
        end
        
        S(t,:) = S(t,:)+reshape(foreground,1,N);
        
    end
end

%  Normalize for imaging purposes
idx = find(S);
values = randn(size(idx));
values = values-min(values);
values = values/max(values);
S(idx) = values;

% ======================= Combined matrix =======================
M = L;
M(idx) = S(idx);


%% Do reconstruction

% ================== Run R2PCA and compute error ==================
fprintf('Running R2PCA... \n');

% [Lhat,Shat] = R2PCA(M,r);

% [L, S, U, T, info] = R2PCA_astronomy(M, r, tol, timelimit, verbose)
tol = 1e-9;
timelimit = false;
verbose = 1;

tic
[Lhat,Shat,Uhat,Chat,info] = R2PCA_astronomy(M, r, tol, timelimit, verbose);
toc

%Error in L
stats.err.l = norm(L-Lhat,'fro')/norm(L,'fro');
fprintf('L Error = %1.1d. \n',stats.err.l);

%Error in S
% (take abs val of Shat, zero any entries <1e-9)
Shat_ = abs(Shat);
Shat_(Shat_<1e-9)=0;
stats.err.s = norm(S-Shat_,'fro')/norm(S,'fro');
fprintf('S Error = %1.1d. \n',stats.err.s);

% (binarize + compute dice score)
stats.dice.s = dice(S>0,Shat_>0);
fprintf('S Dice coeff = %1.1d. \n',stats.dice.s);



%% Make figures of recon results

%%% Display dynamic movie frames of input images and recovered L & S
f = figure('position', [238 556 1101 305],'InvertHardcopy','off','color','w');

savegif = false;
if savegif
    outdir = '/Users/robertjones/Desktop';
    filename = [outdir filesep 'r2pca_astronomy_sim_movie.gif'];
end
% ======================= Display Video =======================
for t=1:frames
    
    % Combination of low-rank background plus sparse foreground
    figure(1);
    subplot(1,3,1);
    imagesc(reshape(M(t,:),height,width)); clim([0 1]);
    colormap(gray);
    title({'M = L + S','(input data)',['(r=' num2str(r) ')']});
    set(gca, 'XTickLabelMode','manual','XTickLabel',[], ....
        'YTickLabelMode','manual','YTickLabel',[], 'FontSize',15);
    drawnow;

    % Low-rank background recovered by R2PCA
    figure(1);
    subplot(1,3,2);
    imagesc(reshape(abs(Lhat(t,:)),height,width));  clim([0 1]);
    colormap(gray);
    title({'L = Blinking stars','(low-rank background)','recovered by R2PCA'});
    set(gca, 'XTickLabelMode','manual','XTickLabel',[], ....
        'YTickLabelMode','manual','YTickLabel',[], 'FontSize',15);
    drawnow;

    % Sparse foreground recovered by R2PCA
    figure(1);
    subplot(1,3,3);
    imagesc(reshape(abs(Shat(t,:)),height,width));  clim([0 1]);
    colormap(gray);
    title({'S = Moving objects','(sparse foreground)','recovered by R2PCA'});
    set(gca, 'XTickLabelMode','manual','XTickLabel',[], ....
        'YTickLabelMode','manual','YTickLabel',[], 'FontSize',15);
    drawnow;

    if savegif
        fr = getframe(1);
        im = frame2im(fr);
        [imind,cm] = rgb2ind(im,256);
        if t == 1
            imwrite(imind,cm,filename,'gif','Loopcount',inf,'DelayTime',0.15);
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0.15);
        end
    else
        pause(.03);
    end
end


pause(1);


f2 = figure('color','w','position',[289 205 990 661],'InvertHardcopy','off');
subplot(3,2,1:2);
imagesc(M); clim([0 1]); set(gca,'colormap',gray);
title('M = L + S'); set(gca,'FontSize',15);

subplot(323);
imagesc(L); clim([0 1]); set(gca,'colormap',gray);
title('L'); set(gca,'FontSize',15);
subplot(325);
imagesc(abs(Lhat)); clim([0 1]); set(gca,'colormap',gray);
title('L hat'); set(gca,'FontSize',15);

subplot(324);
imagesc(S); clim([0 1]); set(gca,'colormap',gray);
title('S'); set(gca,'FontSize',15);
subplot(326);
imagesc(abs(Shat)); clim([0 1]); set(gca,'colormap',gray);
title('S hat'); set(gca,'FontSize',15);

























