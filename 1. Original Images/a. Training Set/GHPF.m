function MOutput = GHPF(MInput);
%Gaussian High Pass Filter


[m, n]=size(MInput);
f_transform=fft2(MInput);
f_shift=fftshift(f_transform);
p=m/2;
q=n/2;
d0=70;

for i=1:m
    for j=1:n
        distance=sqrt((i-p)^2+(j-q)^2);
        low_filter(i,j)=1-exp(-(distance)^2/(2*(d0^2)));
    end
end
filter_apply=f_shift.*low_filter;
image_orignal=ifftshift(filter_apply);
MOutput=abs(ifft2(image_orignal)); 
