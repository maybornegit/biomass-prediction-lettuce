function [images, digits,traits] = prepareDataforTraitsRegression(aDigitTable,path)
imagePaths = iGetFullDigitPaths( aDigitTable );
digitImds = imageDatastore(imagePaths, 'LabelSource', 'foldernames' );
[images, digits] =imds2array(digitImds);
LFW = aDigitTable.LFW;
LDW = aDigitTable.LDW;
LA = aDigitTable.LA;
traits=[LFW LDW LA];
function imagePaths = iGetFullDigitPaths( aDigitTable )
% Base digit path
digitPath =path;
% Add digit folder to path name
imagePaths = strcat(aDigitTable.label, filesep, aDigitTable.image);
% Add full path
imagePaths = cellfun(@(s)strcat(digitPath,s),imagePaths,'UniformOutput',false);
end
end

function [X, T] = imds2array(imds)
imagesCellArray = imds.readall();
numImages = numel( imagesCellArray );
[h, w, c] = size( imagesCellArray{1} );
X = zeros( h, w, c, numImages );
for i=1:numImages
X(:,:,:,i) = im2double( imagesCellArray{i} );
end
T = imds.Labels;
end

