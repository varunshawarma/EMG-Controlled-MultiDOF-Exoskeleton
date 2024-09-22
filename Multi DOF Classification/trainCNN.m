function [cnn_net, accuracy, score] = trainCNN(Kinematics, Features, varargin)
%TRAINCONTINUOUSCNN Summary of this function goes here
%   Detailed explanation goes here

[Kinematics, Features, varargout] = alignTrainingData_jag(Kinematics, Features, 1:192, 'standard');
% Defining States
window = 10;
state = zeros(1, length(Kinematics));
count = 1;

while count <= length(Kinematics)
    state(count) = findState(count, Kinematics);
    % Window
    windowcount = min(count+window,length(Kinematics));
    if findState(windowcount,Kinematics) == state(count)
        state(count:windowcount) = state(count);
        count = windowcount+1;
    else
        count = count+1;
    end
end

classificationArray = state;
windowSize = 3;
for i = 1:length(Features)-windowSize
    Windows{1,i} = classificationArray(i+windowSize-1);
    Windows{2,i} = Features(:,i:i+windowSize-1);
end

labels = [Windows{1,:}];

train_data = zeros(height(features), windowSize, 1, length(Windows));

for i = 1:length(Windows)
    train_data(:,:,1,i) = Windows{2,i};
end

[trainmask, valmask, testmask] = dividerand(length(train_data),0.80, 0.10, 0.10);

train_X = train_data(:,:,:,trainmask);
test_X = train_data(:,:,:,valmask);
train_Y = categorical(labels(trainmask));
test_Y = categorical(labels(valmask));

learnRate = .05;
maxEpoch = 25;
PATIENCE = 8;
inputSize = [height(features),windowSize,1]; %only one channel (i.e. not RGB image)

cnn_options = trainingOptions('sgdm', 'InitialLearnRate', learnRate, 'MaxEpochs', maxEpoch,...
    'Shuffle','every-epoch', 'Plots', 'training-progress', 'ValidationData', {test_X, test_Y},...
    'ValidationFrequency', 30, 'GradientThreshold', 100000, 'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', .5, 'LearnRateDropPeriod', 3, 'ValidationPatience', PATIENCE...
    );
    % ,'Plots','none'...
    % );

% Set up the Neural Network
% layers = [ ...
%     imageInputLayer(inputSize, 'Normalization', 'rescale-zero-one'  )
%     convolution2dLayer([1,3],30)
%     batchNormalizationLayer
%     %     maxPooling2dLayer([2,2])
% 
%     convolution2dLayer([1,3],20)
%     batchNormalizationLayer
%     maxPooling2dLayer([2,2])
% 
%     convolution2dLayer([1,3],10)
%     batchNormalizationLayer
%     maxPooling2dLayer([2,2])
% 
%     convolution2dLayer([1,3],5)
%     batchNormalizationLayer
%     maxPooling2dLayer([2,2])
% 
%     convolution2dLayer([1,1],5)
%     batchNormalizationLayer
%     maxPooling2dLayer([2,2])
% 
%     fullyConnectedLayer(128)
%     leakyReluLayer
%     dropoutLayer(.2)
%     fullyConnectedLayer(64)
%     leakyReluLayer
%     dropoutLayer(.2)
%     fullyConnectedLayer(length(unique(classificationArray)))
%     softmaxLayer
% 
%     classificationLayer];

layers = [ ...
    imageInputLayer(inputSize, 'Normalization', 'rescale-zero-one')
    convolution2dLayer([1,3],30, 'Padding','same')
    batchNormalizationLayer
    maxPooling2dLayer([2,2], 'Padding', 'same')

    convolution2dLayer([1,2],20, 'Padding','same')
    batchNormalizationLayer
    maxPooling2dLayer([2,2], 'Padding', 'same')

    convolution2dLayer([1,1],10, 'Padding','same')
    batchNormalizationLayer
    maxPooling2dLayer([2,2], 'Padding', 'same')

    convolution2dLayer([1,1],5, 'Padding','same')
    batchNormalizationLayer
    maxPooling2dLayer([2,2], 'Padding', 'same')

    fullyConnectedLayer(128)
    leakyReluLayer
    dropoutLayer(.2)
    fullyConnectedLayer(64)
    leakyReluLayer
    dropoutLayer(.2)
    fullyConnectedLayer(length(unique(classificationArray)))
    softmaxLayer

    classificationLayer];

[cnn_net, trainInfo] = trainNetwork(train_X,train_Y,layers,cnn_options);

preds = classify(cnn_net, train_data(:,:,:,testmask))';
score = predict(cnn_net, train_data(:,:,:,testmask))';

testlabels = categorical(labels(testmask));

accuracy = sum(preds==testlabels)/length(testlabels);

% State determination function
    function s = findState(count, Kinematics)
        if all(Kinematics(1:5, count) > 0)
            s = 1; % Grasp
        elseif all(Kinematics(1:5, count) < 0)
            s = 2; % Open
        elseif Kinematics(12, count) > 0
            s = 3; % Pronation
        elseif Kinematics(12, count) < 0
            s = 4; % Supination
        elseif sum(Kinematics(:,count)) == 0        
            s = 0; % Neutral
        else
            s = -1; % Remove
        end
    end
end