clear all;
datadir=uigetdir({},'选择图像数据文件夹');
subdir=dir(datadir);
datadir1='E:\论文\Estimation of above ground biomass for winter wheat using digital images and deep convolutional neural network\试验数据\AGB_4\试验数据';
% type=datadir(end-3:end);
subdir(1:2) = [];
si=length(subdir);
for i =1: si
    if subdir(i).isdir==1
        imgfiles=dir(fullfile(datadir,subdir(i).name));
        imgfiles(1:2) = [];
        for j=1:length(imgfiles)
            img=im2double(imread(fullfile(datadir,subdir(i).name,imgfiles(j).name))); 
            name=imgfiles(j).name;
            lh=length(name);
            markernumber=name(lh-4);
            switch markernumber
                 case '1'
                   imwrite(img,fullfile(datadir1,'训练数据',subdir(i).name,imgfiles(j).name));
                case '2'
                   imwrite(img,fullfile(datadir1,'训练数据',subdir(i).name,imgfiles(j).name)); 
                case '3'
                   imwrite(img,fullfile(datadir1,'测试数据',subdir(i).name,imgfiles(j).name));
            end
        end
    end
end