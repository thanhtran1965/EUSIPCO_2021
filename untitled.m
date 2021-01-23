rng('default')
dataFolder = '/media/thanh/Samsung_T5/BackupData/3rdYear/1.EUSIPCO/23Jan2021/train-20210123T104354Z-001/train/';

imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

trainingNumFiles = 0.7;
[imdsTrain, imdsValidation] = splitEachLabel(imds, trainingNumFiles, 'randomize');

%%
net = googlenet;

%% 
net.Layers(1)
inputSize = net.Layers(1).InputSize;

%% Replace final layers
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

%% Find the names of the two layers to replace
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]

%% 
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

%% 
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%%
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%% Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%% Train Network
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%% To automatically resize the validation images without performing further data augmentation, use an augmented image datastore without specifying any additional preprocessing operations.
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% 
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%
net = trainNetwork(augimdsTrain,lgraph,options);