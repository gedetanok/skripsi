clc;clear;close all;

rgbImg = imread('C:\Users\DMS\Documents\Conference & Journal\Bahan Paper\Diabetic Retinopathy\IDRiD Dataset\Dataset\A. Segmentation\1. Original Images\a. Training Set\IDRiD_03.jpg');

%%
% channel extraction
rChan = rgbImg(:,:,1);
gChan = rgbImg(:,:,2);
bChan = rgbImg(:,:,3);

[M, N] = size(rChan);

% figure(1)
% subplot(2,3,2),imshow(rgbImg),title('Input');
% subplot(2,3,4),imshow(rChan),title('Red');
% subplot(2,3,5),imshow(gChan),title('Green');
% subplot(2,3,6),imshow(bChan),title('Blue');

%%
% RGB to HSL
rNorm = mat2gray(rChan);
gNorm = mat2gray(gChan);
bNorm = mat2gray(bChan);

rgbNorm = cat(3,rNorm,gNorm,bNorm);

hslImg = rgb2hsl(rgbNorm);

% channel extraction
hChan = hslImg(:,:,1);
sChan = hslImg(:,:,2);
lChan = hslImg(:,:,3);

% figure(1)
% subplot(2,3,2),imshow(hslImg),title('HSL Input');
% subplot(2,3,4),imshow(hChan),title('Hue');
% subplot(2,3,5),imshow(sChan),title('Saturation');
% subplot(2,3,6),imshow(lChan),title('Luminance');

%%
% Green Channel Processing

%Filter using BHPF
gFilter = fn_CLAHE(gChan);

% figure(1)
% subplot(1,2,1),imshow(gChan),title('Before');
% subplot(1,2,2),imshow(gFilter),title('After');

%Threshold
% gThresh = binarizeOtsu(gFilter);
% 
% figure(1)
% subplot(1,2,1),imshow(gChan),title('Green Input');
% subplot(1,2,2),imshow(gThresh),title('Threshold');

%%
% Luminance Channel Extraction

%CLAHE
lCLAHE = fn_CLAHE(lChan);
% 
lContrast = ConStrec(lCLAHE);
% 
lFilter = imresize(medfilt2(lContrast,[3 3]),[M N]);
% 
figure(1)
subplot(2,2,1),imshow(lChan),title('L Input');
subplot(2,2,2),imshow(lCLAHE),title('L CLAHE');
subplot(2,2,3),imshow(lContrast),title('L Contrast');
subplot(2,2,4),imshow(lFilter),title('L Median');

figure(2),imshow(imcomplement(lCLAHE)),title('L CLAHE');