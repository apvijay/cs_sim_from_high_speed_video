% Simulate compressive measurements from frames of a high speeed video 
% and reconstruct
% using Scrambled Block Hadamard Ensemble

% 
% 128x128: 10000: grayscale:  Total time   (secs) :   373.8
% 128x128: colour: 
% R: Total time   (secs) :   398.2
% G: Total time   (secs) :   616.5
% B: Total time   (secs) :   433.9

% 128x128: colour: M=10000: frames 3 to 1002: 1100.5, 1662.6, 1130.5

clc;
close all;
clear all;

addpath(genpath('/home/ipcv15/iitm/code/SPGL1/spgl1-master/'));

%% Directory information
for num_frames = [302 902]

my_dir = '/home/ipcv15/iitm/code/CS-MUVI/card_mons/';
imlist = dir(my_dir);
imlist = imlist(3:num_frames); % first two are . and .. directories

%% Parameters
siz = [256 256];
M = (num_frames - 2) * 40; %40000; %prod(siz);
num_color = 3;

%% Measurements structure
meas.siz = siz;

%% Parameters structure
params.dir = my_dir;
params.num_images = length(imlist);
params.img_names = imlist;
params.num_color = num_color;
params.meas_per_frame = ceil(M / params.num_images);

total_meas = params.meas_per_frame * params.num_images;
ymeas = zeros(total_meas, params.num_color);

row_perm = randperm(prod(meas.siz),total_meas);
col_perm = randperm(prod(meas.siz));
% column permutation using LCP
% LCP_A = 29; % should be relative prime with N
% colperm = rem(LCP_A * [0:N-1],N) + 1;
B = 32;

Phifun = @(x) SBHEfun(x,row_perm,col_perm,B);
Phitfun = @(x) SBHEtfun(x,row_perm,col_perm,B);

%% Obtain compressive measurements
for kk = 1:params.num_images
    img = imread([params.dir params.img_names(kk).name]);
    img = double(img) / 255;
    img = imresize(img, meas.siz, 'bilinear');
    
%     if mod(kk, 1) == 0
%         figure; imshow(img); drawnow;
%     end
    if params.num_color == 1
        img = mean(img, 3);
    end
        
    % BxB Hadamard matrix
    WB = hadamard(B);
    
    % ymeas = fwd_comp_meas(img, meas);
    for ii = 1:params.meas_per_frame
        count = (kk-1) * params.meas_per_frame + ii;
        for jj = 1:params.num_color
            this_img = img(:,:,jj);
%             ymeas(count,jj) = Phifun(this_img(:));
            

            rownum = row_perm(count);

            % form the row in the W matrix, permute and inner product with x
            zfront = B * floor((rownum-1)/B);
            phi = [zeros(1,zfront) WB(rem(rownum-1,B)+1,:) zeros(1,prod(meas.siz)-B-zfront)];
            phi = phi(col_perm);
            ymeas(count,jj) = phi * this_img(:) / sqrt(B); % to make phi unit norm


% 
        end
    end
end

%% l1 reconstruction
imgw = zeros(meas.siz(1), meas.siz(2), params.num_color);
imgout = zeros(meas.siz(1), meas.siz(2), params.num_color);

dwtmode('per');
wave.name = 'db4';
wave.level = 6;
wave.siz = [ meas.siz ];
[tmp, wave.Cbook] = wavedec2(randn(wave.siz), wave.level, wave.name);

Psifun = @(x) reverseWavelet(x, wave); % reconstruction
Psitfun = @(s) forwardWavelet(s, wave); % fwd transform

Afun = @(x) Phifun(Psifun(x));
Atfun = @(x) Psitfun(Phitfun(x));

Amodefun = @(x,mode) Awrapperfun(x,mode,Afun,Atfun);

% SPG L1 solver
opts = spgSetParms('verbosity',1);
sigma = 0.1;

for jj = 1:params.num_color
    xtmp = spg_bpdn(Amodefun, ymeas(:,jj), sigma, opts);
    imgw(:,:,jj) = reshape(xtmp, meas.siz);
    imgout(:,:,jj) = reshape(Psifun(imgw(:,:,jj)),meas.siz);
end
% imgout = reshape(Psifun(imgw),meas.siz);
figure, imshow(uint8(255*imgout));

filenam = ['card_frame_3to' num2str(params.num_images+2) '_sbhe_reconst_' ...
    num2str(meas.siz(1)) '_M' num2str(M) '_colour' num2str(params.num_color) '.ppm'];
% imwrite(uint8(imgout*255),'card_frame_3to302_sbhe_reconst_128_M10000_colour.ppm');
imwrite(uint8(imgout*255),filenam);
clear all;
end