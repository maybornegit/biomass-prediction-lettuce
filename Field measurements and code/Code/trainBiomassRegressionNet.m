clear all;clc;
[filename,filepath] = uigetfile('*.xlsx','选择输入xlsx文件');
str=fullfile(filepath,filename);
trainTable = readtable(str);
[images, label,traits] = prepareDataforTraitsRegression(trainTable,filepath );
numTrainImages = numel(label);
idx = randperm(numTrainImages,floor(numTrainImages*0.8));
trainImages=images(:,:,:,idx);
trainTraits=traits(idx,:);
validationImages=images;
validationImages(:,:,:,idx)=[];
validationTraits=traits;
validationTraits(idx,:,:)=[];
% figure
% histogram(trainBiomass)
% axis tight
% ylabel('Counts')
% xlabel('biomass')
layers = [
    imageInputLayer([128 128 3])

    convolution2dLayer(5,32,'Stride',1)
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(5,64,'Padding',1,'Stride',1)
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
  
    convolution2dLayer(5,128,'Padding',1,'Stride',1)
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,256,'Stride',1) %C4
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2) %S4
    convolution2dLayer(5,512,'Stride',1) %C5
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(3)
    regressionLayer];

miniBatchSize  = 128;
validationFrequency = floor(numel(trainTraits)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',300,...
    'InitialLearnRate',1e-2,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',20,...
    'Shuffle','every-epoch',...
    'ValidationData',{validationImages,validationTraits},...
    'ValidationFrequency',validationFrequency,...
    'ValidationPatience',Inf,...
    'Plots','training-progress',...
    'Verbose',false);
net= trainNetwork(trainImages,trainTraits,layers,options);
predictedTraits=predict(net,validationImages);
[fitresult, gof] = createFit(validationTraits, predictedTraits)
savepath = uigetdir({},'选择网络存放文件夹'); 
save(fullfile(savepath,'EstimationofAGB.mat'),'net');
