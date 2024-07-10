clear all;
datadir=uigetdir({},'Ñ¡ÔñÍ¼ÏñÊý¾ÝÎÄ¼þ¼Ð');
subdir=dir(datadir);
subdir(1:2) = [];
si=length(subdir);
for i =1: si
    if subdir(i).isdir==1
        imgfiles=dir(fullfile(datadir,subdir(i).name));
        imgfiles(1:2) = [];
        for j=1:length(imgfiles)
             img=im2double(imread(fullfile(datadir,subdir(i).name,imgfiles(j).name)));    
             imwrite(img,strcat(datadir,'-À©³ä','\','Original','\',imgfiles(j).name));
             [imgrotate{1,1},imgrotate{2,1},imgrotate{3,1},imgrotate{4,1},imgrotate{5,1}]=augByRotating(img);
             [numimgrotate,x]=size(imgrotate);
             for ii= 1:numimgrotate
%              for ii=3:3:numimgrotate
                 imwrite(imgrotate{ii,1},strcat(datadir,'-À©³ä','\',subdir(i).name,'\',num2str(ii*30),'-',imgfiles(j).name));
                 img1=imgrotate{ii,1};             
                 hsv=rgb2hsv(img1);
                 h=hsv(:,:,1);s=hsv(:,:,2);v=hsv(:,:,3);
                 for k=1:2
                    vD=(10-k)/10*v;
                    hsvD=cat(3,h,s,vD); imgD{1,k}=hsv2rgb(hsvD);
                    imwrite(imgD{1,k},strcat(datadir,'-À©³ä','\',subdir(i).name,'\',num2str(ii*30),'-',num2str((10-k)/10),'-',imgfiles(j).name));
                    vI=(10+k)/10*v;
                    hsvI=cat(3,h,s,vI); imgI{1,k}=hsv2rgb(hsvI);
                    imwrite(imgI{1,k},strcat(datadir,'-À©³ä','\',subdir(i).name,'\',num2str(ii*30),'-',num2str((10+k)/10),'-',imgfiles(j).name));
                 end
             end
        end
    end
end