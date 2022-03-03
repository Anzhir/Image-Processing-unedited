%% 
% Moving Average filter , removes Noise , enhances and brighten up the
% image and removes gaussian noise
img=imread('gabbie.jpg');
h=ones(4)/16; % ones(blur)/brightness
imshow(img);
figure(2)
out=imfilter(img , h)
imshow(out)
%%
%First edge detection
image=imread('gabbie.jpg');
h=[0 1 0;1 -4 1;0 1 0];
image=rgb2gray(image);
imshow(image)
out=filter2(h,image);
imshow(out > 50  ,[])
%%
%second sharpening
image=imread('gabbie.jpg');
h=[0 -1 0;-1 5 -1;0 -1 0];
out=imfilter(image,h);
imshow(out)
figure(2);
imshow(image)
%%
%fft of images
%image=rgb2gray(imread('gabbie.jpg'));
im_fft=fft2(img);
im_fft=abs(fftshift(im_fft));
imtool(log(im_fft) , [])
%%
%gauess filters gives blurry without ripples as the circle filter gives
image=rgb2gray(imread('gabbie.jpg'));
filtered=imgaussfilt(image , 0.5);
montage({image,filtered})
title('Original Image (Left) Vs. Gaussian Filtered Image (Right)')
%%
%Edge detection , Uses first and second dvs "gradient"
hx=[-1 0 1;-2 0 2;-1 0 1];
hy=hx';
img=rgb2gray(imread('menna.jpg'));
filtered_x=filter2(hx,img);
filtered_y=filter2(hy , img);
%%
image=rgb2gray(imread('coins.jpg'));
[Gmag, Gdir] = imgradient(image , 'sobel'); %sobel m prewit , other methods
figure
colormap jet
colorbar
imshowpair(Gdir,Gmag, 'montage'); 
title('Gmag (left), Direction(right), using Prewitt method')
%You can also use imgradientxy to get the components in x and y
%%
%threshholding
image=rgb2gray(imread('coins.jpg'));
[Gmag, Gdir] = imgradient(image , 'prewitt'); 
threshold_Mag=abs(Gmag > 200);
Threshhold_angle=(abs(Gdir-30) < 15);
imshowpair(image,threshold_Mag, 'montage'); 
title('Gmag (left), Direction(right), using Prewitt method')
%YOU CAN ALSO USE FSPECIAL
%%
%laplacian of gaussian gives edges that are sick
image=rgb2gray(imread('coins.jpg'));
h3=fspecial('log', [20 20]  ,5);
h10=fspecial('log' , [50 50] , 10);
f3=filter2(h3 , image);
f10=filter2(h10 , image);
%%
image=rgb2gray(imread('sand.jpg'));
edges=edge(image , 'Canny' , [0.6]); % [low , high]
imshow(edges)
%%
img=rgb2gray(imread('triangle.jpg'));
filled_img=imfill(img , 'holes');
binary_img=imbinarize(filled_img);
B = bwboundaries(binary_img,'noholes');
%%
%Load img , find edges , find Boundaries , Plot original img + boundaries
%or binarize , fill , and find boundaries without edges
%lazm tb2a filled white shape
imagergb=imread('mobile.jpg');
image=rgb2gray(imagergb);
edge1=edge(image , 'canny' , [0.7]);
edge1=imfill(edge1 , 'holes');
edge2=edge(image , 'canny' , [0.5]);
edge2=imfill(edge2 , 'holes');
b1=bwboundaries(edge1);
b2=bwboundaries(edge2);
imshow(image)
hold on
for k = 1:length(b1)
   boundary = b1{k};
   plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2)
end
for k = 1:length(b2)
   boundary = b2{k};
   plot(boundary(:,2), boundary(:,1), 'b', 'LineWidth', 2)
end
%%
%trace one boundary , lazm tb2a eswd kolo w line bas white=1 
imgrgb=rgb2gray(imread('shapes.jpg'));
img=edge(imgrgb , 'canny' , [0.9]);
r1 = 43;
c1 = 300;
B = bwtraceboundary(img,[r1 c1],'E',8,500,'clockwise');
imshow(imgrgb)
hold on
plot(B(:,2),B(:,1),'r','LineWidth',2)
%%
imgrgb=imread('xd.jpg');
img=rgb2gray(imgrgb);
img=histeq(img);
edges=edge(img , 'canny' , [0.7]);
imshow(edges)

r1=335;
c1=99;
B=bwtraceboundary(edges,[r1 c1],'N',8,2000,'clockwise');
imshow(imgrgb)
hold on
plot(B(:,2),B(:,1),'r','LineWidth',2)
%%
imgrgb=imread('coins2.jpg');
img=rgb2gray(imgrgb);
edges=edge(img , 'canny' , [0.6])
B=bwboundaries(edges);
imshow(imgrgb)
hold on
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
end
%%
img=rgb2gray(imread('triangle.jpg'));
lines=myhough(img , 0.6 , 10);
img2=imread('soduko.jpg');
figure
lines2=myhough(img2 , 0.6 ,6);
%%
%circular hough
imgrgb=imread('golf.jpg');
img=rgb2gray(imgrgb);
h=fspecial('gaussian' , [50 50] , 5);
img_blurred=filter2(h,img);
[centers, radii, metric] = imfindcircles(img_blurred,[100 500]);
imshow(imgrgb)
hold on 
viscircles(centers, radii,'EdgeColor','b');
%%
imgrgb=imread('pp.jpg');
img=rgb2gray(imgrgb);
level=graythresh(img)
imhist(img)
%
BW = imbinarize(img,level);
figure
imshowpair(img > level*255 ,BW,'montage') %same as (img > (level*255))

%%
img=imread('pp.jpg');
img=rgb2gray(img);
t=adaptthresh(img , 0.6);
imshow(imbinarize(img , t))
%%
img=rgb2gray(imread('golf.jpg'));
fun = @(block_struct) graythresh(block_struct.data);
T=blockproc(img , [200 200] , fun);
T=T*255;

fun2=@(block_struct) imbinarize(block_struct.data);

T=blockproc(img , [200 200] , fun2);
%%
img=rgb2gray(imread('gabbie.jpg'));
graycon(img , 1)
%%
img=rgb2gray(imread('road.jpg'));
L=imsegkmeans(img , 4); %can be used with RGB and GRAY
imshowpair(img , labeloverlay(img , L) , 'montage')
%%
he=imread('road.jpg');
img_lab=rgb2lab(he);
L=img_lab(: , : , 1); %grayscaled img
ab = img_lab(:,:,2:3);
ab = im2single(ab);
nColors = 3;
% repeat the clustering 3 times to avoid local minima
pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);
figure(1)
imshow(pixel_labels,[])
title('Image Labeled by Cluster Index');
mask1 = pixel_labels==1;
cluster1 = he .* uint8(mask1);
figure(2)
imshow(cluster1)
title('Objects in Cluster 1');
mask2 = pixel_labels==2;
cluster2 = he .* uint8(mask2);
figure(3)
imshow(cluster2)
title('Objects in Cluster 2');
mask3 = pixel_labels==3;
cluster3 = he .* uint8(mask3);
figure(4)
imshow(cluster3)
title('Objects in Cluster 3');
figure(5)
imshowpair(he , labeloverlay(he,pixel_labels) , 'montage')
%%
%Regionprops
img=imread('road.jpg');
im=rgb2gray(img);
im=imbinarize(im);
image(bwlabel(im)) %bwlabel return labels for connected boundaries in image
%some processing needed before bwlabel tho
%%
imgrgb=imread('golf.jpg');
img=rgb2gray(imgrgb);
im=imbinarize(img);
im=imerode(im , strel('disk' , 3))
info = regionprops(im,'Boundingbox') ;
imshow(im)
hold on
for k = 1 : length(info)
     BB = info(k).BoundingBox;
     rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2) ;
end
%%
%Active contours
I=rgb2gray(imread('golf.jpg'));
I=imbinarize(I)
mask = zeros(size(I));
mask(25:end-25,25:end-25) = 1;
bw = activecontour(I,mask,600);
imshow(bw)
%%
imgrgb=imread('golf.jpg');
img=rgb2gray(imgrgb);
temp=imread('temp.bmp');
c=normxcorr2(temp , img);
t=0.7;
c=c>t;
figure
imshow(c)
g1=ginput(1);
yoffSet = g1(2)-size(temp,1);
xoffSet = g1(1)-size(temp,2);
figure(2)
imshow(imgrgb)
drawrectangle(gca,'Position',[xoffSet,yoffSet,size(temp,2),size(temp,1)]);
%%
imgrgb=imread('golf.jpg');
img=rgb2gray(imgrgb);
temp=imread('temp.bmp');
c=normxcorr2(temp , img);
t=0.7;
c=c>t;
[m,n]=size(img);
c=imresize(c , [m , n]);
%region props
info = regionprops(c,'Boundingbox') ;
imshow(imgrgb)
hold on
for k = 1 : length(info)
     BB = info(k).BoundingBox;
     rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2) ;
end
%%
imgrgb=imread('golf.jpg');
img=rgb2gray(imgrgb);
temp=imread('temp.bmp');
c=normxcorr2(temp , img);
t=0.7;
c=c>t;
[m,n]=size(img);
c=imresize(c , [m , n]);
imshow(labeloverlay(imgrgb,c))
%%

