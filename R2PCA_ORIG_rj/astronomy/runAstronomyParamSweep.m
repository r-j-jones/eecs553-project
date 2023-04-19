
% ====================================================================
% (Sample code to replicate the astronomy experiments)
% Run astronomy simulations for different values of :
%    r, numStars, maxSpeed, numObjects, objectWidth
%
%
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% =====================================================================

% clear; 
% close all; 
% clc;

warning ('off','all'); 
rng('shuffle');

%% Set parameters for simulations

% ======================================================================
% =========================== GENERAL SETUP ============================
% ======================================================================
%%% %%% FIXED VARIABLES %%% %%%
height = 90;        % height of each video frame
width = 120;        % width of each video frame
frames = 100;       % number of frames in the video

% %%% Test variables to vary
% r = 5;              % rank of the background
% numStars = 500;     % Number of blinking "stars" in the background. 
%                     % In the paper we play with a range between 10 and 10800. 
%                     % This determines how coherent is the background. 
%                     % The smaller NumStars, the more coherent.
% maxSpeed = 5;       % How fast each object will be moving 
% numObjects = 100;   % Number of objects that will be moving around. 
%                     % In the paper we play with a range between 1 and 400.
% objectWidth = 5;    % Width of each object (in number of pixels)
params.r = [1,5,10];
params.numStars = [10,100,1000,10000];
params.numObjects = [5,10,100];
params.objectWidth = [1,5,10];
params.maxSpeed = [1,5,10];
params.numTrials = 10;

results.error.l = zeros(length(params.r), length(params.numStars), length(params.numObjects), ...
    length(params.objectWidth), length(params.maxSpeed), params.numTrials);
results.error.s = zeros(length(params.r), length(params.numStars), length(params.numObjects), ...
    length(params.objectWidth), length(params.maxSpeed), params.numTrials);
results.dice.s = zeros(length(params.r), length(params.numStars), length(params.numObjects), ...
    length(params.objectWidth), length(params.maxSpeed), params.numTrials);

results.time = zeros(length(params.r), length(params.numStars), length(params.numObjects), ...
    length(params.objectWidth), length(params.maxSpeed), params.numTrials);

results.lor.time = zeros(length(params.r), length(params.numStars), length(params.numObjects), ...
    length(params.objectWidth), length(params.maxSpeed), params.numTrials);
results.lor.itr = zeros(length(params.r), length(params.numStars), length(params.numObjects), ...
    length(params.objectWidth), length(params.maxSpeed), params.numTrials);

results.sp.time = zeros(length(params.r), length(params.numStars), length(params.numObjects), ...
    length(params.objectWidth), length(params.maxSpeed), params.numTrials);
results.sp.itr = zeros(length(params.r), length(params.numStars), length(params.numObjects), ...
    length(params.objectWidth), length(params.maxSpeed), params.numTrials);


%R2PCA parameters to use for all tests
tol = 1e-9;
timelimit = false;
verbose = false;

for tInd = 1:params.numTrials
    for rInd=1:length(params.r)
        r = params.r(rInd);
        for nsInd=1:length(params.numStars)
            numStars = params.numStars(nsInd);
            for owInd=1:length(params.objectWidth)
                objectWidth = params.objectWidth(owInd);
                for noInd=1:length(params.numObjects)
                    numObjects = params.numObjects(noInd);
                    for msInd=1:length(params.maxSpeed)
                        maxSpeed = params.maxSpeed(msInd);

                        % ========== Low-rank matrix (background with blinking stars) ==========
                        % fprintf('Creating low-rank matrix with blinking stars... \n');
                        dim = height*width;       % Ambient dimension
                        U = abs(randn(dim,r));    % Basis of r-dimensional subspace
                        stars = randsample(dim,min(numStars,dim));  % Location of stars
                        U(stars,:) = 100*abs(randn(min(numStars,dim),r)); 
                            %basis has more energy on rows corresponding to "stars" pixels
                        
                        Theta = abs(randn(r,frames));   % Coefficients of each frame
                                                        % wrt the low-rank basis
                        L = U*Theta;                    % Low-rank matrix
                        
                        %At this point, each frame is a column and each row is a pixel. 
                        % We transpose everything to speed things up.
                        L = L';
                        N = height*width;
                        dim = frames;
                        L = L/max(L(:));  %normalize for imaging purposes.
                        
                        % =============== Sparse matrix (moving objects) ===============
                        fprintf('Creating sparse matrix with moving objects... \n');
                        [S, idx] = generateS(dim, N, height, width, numObjects, maxSpeed, objectWidth);
                        
                        % ======================= Combined matrix =======================
                        M = L;
                        M(idx) = S(idx);
                        
                        % ================== Run R2PCA and compute error ==================
                        fprintf('Running R2PCA... \n');
                        
                        % [Lhat,Shat] = R2PCA(M,r);
                        
                        % [L, S, U, T, info] = R2PCA_astronomy(M, r, tol, timelimit, verbose)
                        
                        
                        itertic=tic;
                        [Lhat,Shat,Uhat,Chat,info] = R2PCA_astronomy(M, r, tol, timelimit, verbose);
                        itertoc=toc(itertic);

%                         results.error.l = zeros(length(params.r), length(params.numStars), length(params.numObjects), ...
%                           length(params.objectWidth), length(params.maxSpeed), params.numTrials);

                        err_l = norm(L-Lhat,'fro')/norm(L,'fro');
                        results.error.l(rInd,nsInd,noInd,owInd,nsInd,tInd) = err_l;
                        
                        Shat_ = abs(Shat);
                        Shat_(Shat_<tol)=0;
                        err_s = norm(S-Shat_,'fro')/norm(S,'fro');
                        results.error.s(rInd,nsInd,noInd,owInd,nsInd,tInd) = err_s; 
                        
                        dice_s = dice(S>0,Shat_>0);
                        results.dice.s(rInd,nsInd,noInd,owInd,nsInd,tInd) = dice_s; 
                        
                        results.time(rInd,nsInd,noInd,owInd,nsInd,tInd) = itertoc; 
                        
                        results.lor.time(rInd,nsInd,noInd,owInd,nsInd,tInd) = info.U.elapTime;
                        results.lor.itr(rInd,nsInd,noInd,owInd,nsInd,tInd) = info.U.itr;
                        
                        results.sp.time(rInd,nsInd,noInd,owInd,nsInd,tInd) = info.T.elapTime;
                        results.sp.itr(rInd,nsInd,noInd,owInd,nsInd,tInd) = info.T.itr;

                    end
                end
            end
        end
    end
end




















function [S, idx] = generateS(dim, N, height, width, numObjects, maxSpeed, objectWidth )

S = zeros(dim,N);
for no = 1:numObjects
    %==Create Starting Point and Direction of Moving Object==
    %Will it start from top, bottom, left or right?
    tblr = randi(4);
    switch tblr
        case 1
            object.x0 = randi(width);
            object.y0 = height;
            object.deltax = sign(randn)*randi(maxSpeed);
            object.deltay = -randi(maxSpeed);
        case 2
            object.x0 = randi(width);
            object.y0 = 1;
            object.deltax = sign(randn)*randi(maxSpeed);
            object.deltay = randi(maxSpeed);
        case 3
            object.x0 = 1;
            object.y0 = randi(height);
            object.deltax = randi(maxSpeed);
            object.deltay = sign(randn)*randi(maxSpeed);
        case 4
            object.x0 = width;
            object.y0 = randi(height);
            object.deltax = -randi(maxSpeed);
            object.deltay = sign(randn)*randi(maxSpeed);
    end
    
    %Put object in frames, and vectorize frames to form S
    object.startingTime = randi(dim);
    for t=1:dim
        object.x = object.x0 + object.deltax*(t-object.startingTime);
        object.y = object.y0 + object.deltay*(t-object.startingTime);
        
        foreground = zeros(height,width);
        
        if object.x>=1 && object.x<=width && object.y>=1 && object.y<=height
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

end



