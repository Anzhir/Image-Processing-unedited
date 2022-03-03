function lines = myhough(img , canny_threshold , npeaks )
%Takes RGB IMAGES AND OUTPUTS THE LINES AND PLOTS THEM ON IT
image=img;
edges=edge(image , 'canny' , [canny_threshold]);
%Hough
[H,T,R] = hough(edges);
%Hough Peaks
peaks = houghpeaks(H,npeaks);
%Hough Lines
lines = houghlines(image,T,R,peaks);
figure
imshow(img)
hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end




end

