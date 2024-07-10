clear all;
datadir=uigetdir({},'选择图像数据文件夹');
subdir=dir(datadir);
subdir(1:2) = [];
si=length(subdir);
for i =1:si
    if subdir(i).isdir==1
        imgfiles=dir(fullfile(datadir,subdir(i).name));
        imgfiles(1:2) = [];
        for j=1:length(imgfiles)
            results{i,1}{j,1}=imgfiles(j).name;
            results{i,1}{j,2}=subdir(i).name;
        end
    else
        subdir(i)=[];
    end
end
num=size(results);
Fresult=results{1,1}(:,1:2);
for ii =2:num(1,1) 
gt=results{ii,1}(:,1:2);
Fresult=[Fresult;gt];
end
X=xlsread('E:\论文\Estimation of above ground biomass for winter wheat using digital images and deep convolutional neural network\试验数据\AGB_4\Biomass.xlsx', 'Training');
Biomass=prepareBiomass(X,25);
[filename,filepath]=uigetfile('*.xlsx', '输出xlsx文件');
output=fullfile(filepath,filename);
xlswrite(output,Fresult,'Training', 'A2');
xlswrite(output,Biomass,'Training', 'C2');

