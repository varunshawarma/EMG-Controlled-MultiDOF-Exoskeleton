clear;clc;close all;
addpath(genpath("../../../Students/Grads/CDO/CustomMatlabFunctions"))
addpath("D:\CJT\HAPTIX Offline\")
addpath("\\Neurorobotrt2\c\Users\Administrator\Box\NeuroRoboticsLab\JAGLAB\Projects\Adaptive EMG control")
addpath("D:\Multi DOF Classification\")

files = {"D:\SmartHome\PvNP_Wrist_Forearm\S1_P\TaskData_20230308-174844.kdf", 
    "D:\SmartHome\PvNP_Wrist_Forearm\S1_NP\TaskData_20230308-171058.kdf",
    "D:\SmartHome\PvNP_Wrist_Forearm\S2_P\TaskData_20230310-160842.kdf",
    "D:\SmartHome\PvNP_Wrist_Forearm\S2_NP\TaskData_20230310-152716.kdf",
    "D:\SmartHome\PvNP_Wrist_Forearm\S3_P\TaskData_20230313-112317.kdf",
    "D:\SmartHome\PvNP_Wrist_Forearm\S3_NP\TaskData_20230313-105333.kdf"};

file_path = files{1};

% [Kinematics, Features,~,~,NIPTime] = readKDF(file_path);

[states, features, idxs, state] = preprocessData(file_path);
%%
classificationArray = states;
windowSize = 3;
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

%%
clc;
% Use the first example from the training set
inputExample = train_data(:,:,:,1);

% Class index of the predicted class
classIdx = preds(1);

% Define the size of the occlusion mask
maskSize = 5;

% Call the occlusion sensitivity function
sensitivityMap = occlusionSensitivity(cnn_net, inputExample, classIdx, maskSize);

% Visualize the input image
figure;
imshow(inputExample, []);
hold on;

% Overlay the sensitivity map
imagesc(sensitivityMap, 'AlphaData', 0.5);  % Adjust AlphaData for transparency
colormap jet;
colorbar;
title('Occlusion Sensitivity Map');

%% Complete Feature Map
clc;
% Example usage:
sampleInput = test_X(:,:,:,1);  % Select a sample input from the test data
layerNames = {'conv_1', 'conv_2'};  % Specify the layers you want to visualize
plotSummedFeatureMaps(cnn_net, sampleInput, layerNames); 

%% Visualize Feature Maps
sampleInput = test_X(:,:,:,1);  % Select a sample input from the test data
plotFeatureMaps(cnn_net, sampleInput, 'conv_1', 30);  % Plot feature maps from the first conv layer
plotFeatureMaps(cnn_net, sampleInput, 'conv_2', 20);  % Plot feature maps from the second conv layer
%%
function plotFeatureMaps(net, inputImage, layerName, numFeatureMaps)
    % Get the activations from the specified layer
    act = activations(net, inputImage, layerName);
    
    % Determine the number of rows and columns for subplots
    numRows = ceil(sqrt(numFeatureMaps));
    numCols = ceil(numFeatureMaps / numRows);
    
    % Plot the feature maps
    figure;
    for i = 1:numFeatureMaps
        subplot(numRows, numCols, i);
        imagesc(act(:,:,i));
        title(['Feature Map ', num2str(i)], 'FontSize', 10);
        colorbar;
        set(gca, 'FontSize', 10);  % Increase font size for better readability
        caxis([min(act(:)), max(act(:))]);  % Set consistent color limits
        axis off;  % Turn off axis for better visualization
    end
    sgtitle(['Feature Maps from Layer: ', layerName], 'FontSize', 14);  % Overall title with increased font size
end

function sensitivityMap = occlusionSensitivity(net, inputImage, classIdx, maskSize)
    [height, width, channels] = size(inputImage);
    sensitivityMap = zeros(height, width);
    
    % Convert input to 4-D array for single image
    dlInput = dlarray(single(inputImage), 'SSC');
    dlInput = dlInput(:,:,:,1);  % Ensure it's a 4-D array

    % Get the baseline score for the classIdx
    scores = predict(net, dlInput);
    baselineScore = scores(classIdx);

    % Slide the occlusion mask over the input image
    for i = 1:height-maskSize+1
        for j = 1:width-maskSize+1
            occludedImage = inputImage;
            occludedImage(i:i+maskSize-1, j:j+maskSize-1, :) = 0;  % Apply the occlusion mask

            % Convert to 4-D array for single image and predict
            dlOccludedInput = dlarray(single(occludedImage), 'SSC');
            dlOccludedInput = dlOccludedInput(:,:,:,1);  % Ensure it's a 4-D array
            scores = predict(net, dlOccludedInput);
            occludedScore = scores(classIdx);

            % Calculate the sensitivity (drop in score)
            sensitivityMap(i, j) = baselineScore - occludedScore;
        end
    end

    % Normalize the sensitivity map
    sensitivityMap = sensitivityMap - min(sensitivityMap(:));
    sensitivityMap = sensitivityMap / max(sensitivityMap(:));
end

function plotSummedFeatureMaps(net, inputImage, layerNames)
    % Check if inputImage is correctly formatted as a 4-D array
    if ndims(inputImage) ~= 4
        inputImage = reshape(inputImage, [size(inputImage, 1), size(inputImage, 2), size(inputImage, 3), 1]);
    end

    % Extract and sum activations from multiple layers
    figure;
    numLayers = length(layerNames);
    for idx = 1:numLayers
        % Get the activations from the specified layer
        act = activations(net, inputImage, layerNames{idx});
        
        % Sum the activations across all feature maps
        summedActivations = sum(act, 3);
        
        % Plot the summed activations
        subplot(1, numLayers, idx);
        imagesc(summedActivations);
        title(['Summed Feature Maps from ', layerNames{idx}], 'FontSize', 10);
        colorbar;
        set(gca, 'FontSize', 10);  % Increase font size for better readability
        axis on;  % Turn on axis for better visualization
    end
    sgtitle('Summed Feature Maps Across Layers', 'FontSize', 14);  % Overall title with increased font size
end



