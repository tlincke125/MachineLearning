function filted=cov_filter(image,sizeof,overlap)
[im_rows,im_cols]=size(image);
filted=zeros(ceil(im_rows/sizeof),ceil(im_cols/sizeof),2);
r=1;
for i=1:sizeof-overlap:im_rows-sizeof
    c=1;
   for j=1:sizeof-overlap:im_cols-sizeof
     dy=diff(image(i:i+sizeof-1,j:j+sizeof-1),1,1);
     dx=diff(image(i:i+sizeof-1,j:j+sizeof-1),1,2);
     [evec,eval]=eig(cov(dx,dy));
     ee=evec*eval;
     filted(r,c,:)=ee(:,2);    
     c=c+1;
   end
    r=r+1;
end


end