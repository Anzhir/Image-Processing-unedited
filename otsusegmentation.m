function RGB = ostusegmentation(imgrgb , nlevels)
%ostu is a histogram thresholding algorithm , uses minimization of m1 , m2
%and tries to minimize distance to mg from all clusters
img=rgb2gray(imgrgb);
thresh=multithresh(img , nlevels);
quantized=imquantize(img , thresh);
RGB=label2rgb(quantized);
figure
imshow(quantized , [])
figure
imhist(img)
figure
imshowpair(imgrgb , RGB , 'montage');


end

