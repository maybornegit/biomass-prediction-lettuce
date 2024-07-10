function [img90,img180,img270,imgflr,imgfud] = augByRotating(img)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
img=im2double(img);
img90=imrotate(img,90);
img180=imrotate(img,180);
img270=imrotate(img,270);
imgflr=fliplr(img);
imgfud=flipud(img);
end

