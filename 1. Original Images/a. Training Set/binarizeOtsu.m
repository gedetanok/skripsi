function imOut = binarizeOtsu(img)

level = graythresh(img);
imOut = im2bw(img,level);