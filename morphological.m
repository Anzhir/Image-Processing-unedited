%%
%Boundary Extraction with Morphology
%boundary=A-(AerosionB);
img=rgb2gray(imread('golf.jpg'));
img=imbinarize(img);
s=strel('square' , 5);
boundary=img - imerode(img , s);
imshowpair(img , boundary , 'montage')
%%
img=imread('4.jpg');
otsusegmentation(img , 3 );