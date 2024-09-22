function [accuracy, testlabels, preds, cnn_net, score, CpredT] = trainAgnosticContinuousCNN(features, classificationArray, windowSize, varargin)
% Inputs:
%   features            - MxN matrix of EMG features, where M is the number of features (channels)
%
%   classificationArray - 1xN vector of class labels corresponding to each data point
%
%   windowSize          - Integer specifying the size of the window for segmenting the data.
%
%   varargin            - Optional parameter for additional title in confusion matrix plot.

% Outputs:
%   accuracy            - Scalar value that represents the accuracy of the CNN model on the test set 
%               
%   testlabels          - Categorical array (1xP) of true class labels for the test set, where P is the number of test data points
%                
%   preds               - Categorical array (1xP) of predicted class labels by the CNN for the test set.
%
%   cnn_net             - The trained convolutional neural network (CNN) object.
%   
%   score               - Matrix (P x C) of prediction scores, where P is the number of test data points and C is the number of prediction classes.
%
%   CpredT              - 1x500 vector of prediction times for 500 test samples.

for i = 1:length(features)-windowSize
    % if classificationArray(i+windowSize-1) == 0
    %     Windows{1,i} = "RST";
    % else
    Windows{1,i} = classificationArray(i+windowSize-1);
    % end
    Windows{2,i} = features(:,i:i+windowSize-1);
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

CpredT = zeros(1,500);
for i = 1:500
    tic;
    singleP = classify(cnn_net, test_X(:,:,:,i));  % Predicting only the first sample
    CpredT(1,i) = toc;
end

preds = classify(cnn_net, train_data(:,:,:,testmask))';
score = predict(cnn_net, train_data(:,:,:,testmask))';

testlabels = categorical(labels(testmask));

accuracy = sum(preds==testlabels)/length(testlabels);

if nargin > 3
    title = sprintf("%s: %f'%' accuracy", varargin{1}, round(accuracy*100));
else
    title = sprintf("CNN Accuracy = %f'%'", round(accuracy*100));
end

if nargin > 3
    figure();
    confusionchart(testlabels, preds, 'Title', title);
end

end