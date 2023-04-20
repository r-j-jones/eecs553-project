function [ images ] = load_validation_data(fname)

vidDir = '/Users/robertjones/Desktop/W23/553/project/data/bmc';
vidPath = [vidDir filesep fname '_gt.mp4'];
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

