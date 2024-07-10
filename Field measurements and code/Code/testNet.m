clear all;
[filename,filepath]=uigetfile('*.mat', '选择导入的网络');
f=fullfile(filepath,filename);
net=importdata(f);
[filename,filepath] = uigetfile('*.xlsx','选择测试xlsx文件');
str=fullfile(filepath,filename);
trainTable = readtable(str);
[testImages, label, testtraits] = prepareDataforTraitsRegression(trainTable,filepath);
predictedtraits=predict(net,testImages);
% num1=numel(testtraits)/3;
% for i =1:num1
%     avgtesttraits(i,1)=(testtraits(1+(i-1)*3,1) +testtraits(2+(i-1)*3,1) +testtraits(3+(i-1)*3,1))/3;
%     avgpredictedtraits(i,1)=(predictedtraits(1+(i-1)*3,1) +predictedtraits(2+(i-1)*3,1) +predictedtraits(3+(i-1)*3,1))/3;
% end
% [fitresult, gof] = createFit(avgtesttraits, avgpredictedtraits)
% 
% [fitresult, gof] = createFit(testtraits, predictedtraits)
testLFW=testtraits(:,1);
predictedLFW=predictedtraits(:,1);
num1=numel(testLFW)/3;
for i =1:num1
    avgtestLFW(i,1)=(testLFW(1+(i-1)*3,1) +testLFW(2+(i-1)*3,1) +testLFW(3+(i-1)*3,1))/3;
    avgpredictedLFW(i,1)=(predictedLFW(1+(i-1)*3,1) +predictedLFW(2+(i-1)*3,1) +predictedLFW(3+(i-1)*3,1))/3;
end
[fitresult, gof] = createFit(avgtestLFW, avgpredictedLFW)

[fitresult, gof] = createFit(testLFW, predictedLFW)

testLDW=testtraits(:,2);
predictedLDW=predictedtraits(:,2);
num2=numel(testLDW)/3;
for i =1:num2
    avgtestLDW(i,1)=(testLDW(1+(i-1)*3,1) +testLDW(2+(i-1)*3,1) +testLDW(3+(i-1)*3,1))/3;
    avgpredictedLDW(i,1)=(predictedLDW(1+(i-1)*3,1) +predictedLDW(2+(i-1)*3,1) +predictedLDW(3+(i-1)*3,1))/3;
end
[fitresult, gof] = createFit(avgtestLDW, avgpredictedLDW)

[fitresult, gof] = createFit(testLDW, predictedLDW)
testLA=testtraits(:,3);
predictedLA=predictedtraits(:,3);
num3=numel(testLA)/3;
for i =1:num3
    avgtestLA(i,1)=(testLA(1+(i-1)*3,1) +testLA(2+(i-1)*3,1) +testLA(3+(i-1)*3,1))/3;
    avgpredictedLA(i,1)=(predictedLA(1+(i-1)*3,1) +predictedLA(2+(i-1)*3,1) +predictedLA(3+(i-1)*3,1))/3;
end
[fitresult, gof] = createFit(avgtestLA, avgpredictedLA)

[fitresult, gof] = createFit(testLA, predictedLA)
