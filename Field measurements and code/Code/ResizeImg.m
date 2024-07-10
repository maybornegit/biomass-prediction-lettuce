clear all;
num=input('请输入调整后图像尺寸（AlexNet：227；VGG16：224）:')
datadir=uigetdir({},'选择图像数据文件夹');
imagesize=[num,num];
subdir=dir(datadir);
subdir(1:2) = [];
si=length(subdir);
% Fresults=cell(si,1);
    for i =1: si
        if subdir(i).isdir==1
            imgfiles=dir(fullfile(datadir,subdir(i).name));
            imgfiles(1:2) = [];
            for j=1:length(imgfiles)
                img=im2double(imread(fullfile(datadir,subdir(i).name,imgfiles(j).name)));
                img=imresize(img,imagesize);
                imwrite(img,strcat(datadir,'\',subdir(i).name,'\',imgfiles(j).name));
            end
        end
    end
    