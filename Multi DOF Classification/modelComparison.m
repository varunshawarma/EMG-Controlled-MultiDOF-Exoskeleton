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
[Kinematics, Features,~,~,NIPTime] = readKDF(file_path);
[states, features, idxs, state] = preprocessData(file_path);

% Model Training and Data Collection
% Initialize arrays to store metrics for each dataset
numDatasets = 6;
numSamples = 500;
numClasses = 5;

% Accuracy
KnnAccuracies = zeros(1, length(files));
CnnAccuracies = zeros(1, length(files));

% Time
KnnPredictionTimes = zeros(numDatasets, numSamples);  
CnnPredictionTimes = zeros(numDatasets, numSamples);

% Confusion Matrix
aggConfMatKNN = zeros(numClasses, numClasses);
aggConfMatCNN = zeros(numClasses, numClasses);


% Loop through each dataset file
for k = 1:length(files)
    file_path = files{k};  % Get the path to the current dataset

    % Data Preprocessing
    [states, features] = preprocessData(file_path);
    
    % Train and test KNN model
    [knnModel, overallAccuracy, cvAccuracy, score, predictT, Kpreds, Ktest] = knnClassifier(states, features);
    KnnAccuracies(1,k) = cvAccuracy;
    KnnPredictionTimes(k,:) = predictT;
    confMatKNN = confusionmat(Ktest, Kpreds);
    aggConfMatKNN = aggConfMatKNN + confMatKNN;

    
    % Train and test CNN model
    [CNNaccuracy, testlabels, CNNpreds, cnn_net, CNNscore, predTime] = trainAgnosticContinuousCNN(features, states, 3);
    CnnAccuracies(1,k) = CNNaccuracy;
    CnnPredictionTimes(k,:) = predTime;
    confMatCNN = confusionmat(testlabels, CNNpreds);
    aggConfMatCNN = aggConfMatCNN + confMatCNN;
end

% Save data to a .mat file
save('ModelPerformance.mat', 'KnnAccuracies', 'CnnAccuracies', 'KnnPredictionTimes', 'CnnPredictionTimes');

%% Accuracy Boxplot
% Create the boxplot for KNN and CNN Accuracies
fig = figure;  
set(fig, 'renderer', 'painters');
boxData = [KnnAccuracies; CnnAccuracies]';
boxplot(boxData, 'Labels', {'KNN', 'CNN'});
title('Accuracy Analysis of KNN vs. CNN on Multiple Datasets');
ylabel('Accuracy (%)');
xlabel('Model');

% Improve the aesthetics
set(gca, 'FontSize', 12);  % Set font size for better visibility 

% Customizing boxplot colors and line styles
lines = [1 0 0; 0 0 1]; % Blue for KNN, Red for CNN
boxes = findobj(gca, 'Tag', 'Box');
medians = findobj(gca, 'Tag', 'Median');
numGroups = size(boxData, 2); % Number of groups
numBoxes = size(boxData, 1) / numGroups; % Number of boxes per group

for i = 1:numBoxes
    patch(get(boxes(2*i-1), 'XData'), get(boxes(2*i-1), 'YData'), lines(1,:), 'FaceAlpha', .5);
    set(boxes(2*i-1), 'LineWidth', 2, 'Color', lines(1,:)); % Blue for KNN
    patch(get(boxes(2*i), 'XData'), get(boxes(2*i), 'YData'), lines(2,:), 'FaceAlpha', .5);
    set(boxes(2*i), 'LineWidth', 2, 'Color', lines(2,:)); % Red for CNN

    set(medians(i*2-1), 'Color', 'k', 'LineWidth', 2); % Black, thick line for KNN median
    set(medians(i*2), 'Color', 'k', 'LineWidth', 2); % Black, thick line for CNN median
end

%% Paired T-Testing for Accuracy
% Step 2: Check for outliers
isOutlierKnn = isoutlier(KnnAccuracies, 'quartiles');
isOutlierCnn = isoutlier(CnnAccuracies, 'quartiles');

% Remove outliers for the sake of demonstration
KnnAccuraciesClean = KnnAccuracies;
CnnAccuraciesClean = CnnAccuracies;

% Step 3: Check for parametric distribution
isNonParametricKnn = adtest(KnnAccuraciesClean);
isNonParametricCnn = adtest(CnnAccuraciesClean);

% Step 4: Choose the statistical test
if ~isNonParametricKnn && ~isNonParametricCnn
    % Both are parametric
    [~, p] = ttest(KnnAccuraciesClean, CnnAccuraciesClean);
    testUsed = 'Paired t-test';
else
    % At least one is nonparametric
    [p,~] = signrank(KnnAccuraciesClean, CnnAccuraciesClean);
    testUsed = 'Wilcoxon Signed Rank Test';
end

% Display results
fprintf('Test used: %s\n', testUsed);
fprintf('P-value: %.4f\n', p);
if p < 0.05
    fprintf('There is a statistically significant difference between the two sets of accuracies.\n');
else
    fprintf('There is no statistically significant difference between the two sets of accuracies.\n');
end

meanKnn = mean(KnnAccuraciesClean);
stdKnn = std(KnnAccuraciesClean);
meanCnn = mean(CnnAccuraciesClean);
stdCnn = std(CnnAccuraciesClean);
fprintf('\nMean and Standard Deviation for KNN Accuracies: Mean = %.4f, Std = %.4f\n', meanKnn, stdKnn);
fprintf('Mean and Standard Deviation for CNN Accuracies: Mean = %.4f, Std = %.4f\n', meanCnn, stdCnn);

%% Paired T-testing for Time
% Aggregate prediction times across all samples for each dataset
meanKnnTimes = mean(KnnPredictionTimes, 2);
meanCnnTimes = mean(CnnPredictionTimes, 2);

% Check for parametric distribution
isNonParametricKnn = adtest(meanKnnTimes);
isNonParametricCnn = adtest(meanCnnTimes);

% Choose the statistical test
if ~isNonParametricKnn && ~isNonParametricCnn
    % Both are parametric
    [~, p] = ttest(meanKnnTimes, meanCnnTimes);
    testUsed = 'Paired t-test';
else
    % At least one is nonparametric
    [p,~] = signrank(meanKnnTimes, meanCnnTimes);
    testUsed = 'Wilcoxon Signed Rank Test';
end

meanKnn = mean(meanKnnTimes);
stdKnn = std(meanKnnTimes);
meanCnn = mean(meanCnnTimes);
stdCnn = std(meanCnnTimes);

% Display results
fprintf('Test used: %s\n', testUsed);
fprintf('P-value: %.4f\n', p);
if p < 0.05
    fprintf('There is a statistically significant difference in prediction times between the two models.\n');
else
    fprintf('No statistically significant difference in prediction times between the two models.\n');
end

fprintf('\nMean and Standard Deviation for KNN Times: Mean = %.4f, Std = %.4f\n', meanKnn, stdKnn);
fprintf('Mean and Standard Deviation for CNN Times: Mean = %.4f, Std = %.4f\n', meanCnn, stdCnn);

%% Time Boxplot
% Calculate Mean Prediction Times
% Initialize arrays to hold mean times
meanKnnTimes = zeros(1, numDatasets);
meanCnnTimes = zeros(1, numDatasets);

% Calculate mean times for each dataset
for k = 1:numDatasets
    meanKnnTimes(k) = mean(KnnPredictionTimes(k, :)) * 1000;  % Convert to milliseconds
    meanCnnTimes(k) = mean(CnnPredictionTimes(k, :)) * 1000;  % Convert to milliseconds
end

% Plot the boxplot
fig = figure;  
set(fig, 'renderer', 'painters');
boxData = [meanKnnTimes; meanCnnTimes]';  % Ensure to update boxData for coloring steps
boxplot(boxData, 'Labels', {'KNN', 'CNN'});
title('Time Analysis of KNN vs. CNN on Multiple Datasets');
ylabel('Mean Prediction Time (milliseconds)');  % Update label to milliseconds
xlabel('Model');

% Enhance aesthetics
set(gca, 'FontSize', 12);  % Set font size for better visibility

% Customizing boxplot colors and line styles
lines = [1 0 0; 0 0 1]; % Blue for KNN, Red for CNN
boxes = findobj(gca, 'Tag', 'Box');
numGroups = size(boxData, 2); % Number of groups
numBoxes = size(boxData, 1) / numGroups; % Number of boxes per group

for i = 1:numBoxes
    patch(get(boxes(2*i-1), 'XData'), get(boxes(2*i-1), 'YData'), lines(1,:), 'FaceAlpha', .5);
    set(boxes(2*i-1), 'LineWidth', 2, 'Color', lines(1,:)); % Blue for KNN
    patch(get(boxes(2*i), 'XData'), get(boxes(2*i), 'YData'), lines(2,:), 'FaceAlpha', .5);
    set(boxes(2*i), 'LineWidth', 2, 'Color', lines(2,:)); % Red for CNN
end




%% Confusion Matrix 
% Define the class labels as strings from '0' to '4'
classLabels = arrayfun(@num2str, 0:4, 'UniformOutput', false);

% Confusion Matrix for KNN
fig = figure;  
set(fig, 'renderer', 'painters');
% Create the confusion chart with row normalization
knnChart = confusionchart(aggConfMatKNN, classLabels, ...
    'Title', 'Normalized Confusion Matrix - KNN', ...
    'Normalization', 'row-normalized');
knnChart.Title = 'Normalized Confusion Matrix for KNN';
% Customize the text color and ensure zeros appear
knnChart.FontColor = 'k';  % Set text color to black for visibility
knnChart.FontSize = 16;    % Increase font size to 14

% Confusion Matrix for CNN
fig = figure;  
set(fig, 'renderer', 'painters');
% Create the confusion chart with row normalization
cnnChart = confusionchart(aggConfMatCNN, classLabels, ...
    'Title', 'Normalized Confusion Matrix - CNN', ...
    'Normalization', 'row-normalized');
cnnChart.Title = 'Normalized Confusion Matrix for CNN';
% Customize the text color and ensure zeros appear
cnnChart.FontColor = 'k';  % Set text color to black for visibility
cnnChart.FontSize = 16;    % Increase font size to 14

%% ROC Curve
clc;

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

[predKNN, scoresKNN] = predict(knnModel, features');  % Check the input and output of predict
predCNN = classify(cnn_net, train_data);
scoresCNN = predict(cnn_net, train_data);

%labels = double(testlabels)-1;
Knnlabels = states(2:end);
Cnnlabels = states(4:end);
AUCknn = zeros(1, 5);
AUCcnn = zeros(1, 5);
%
% Prepare figure
fig = figure;  
set(fig, 'renderer', 'painters');
hold on;

% Colors for plotting
colors = lines(10);  % Get distinct colors for up to 10 classes

% Plot handles for legend
hPlots = [];  % Initialize an empty array to store plot handles
legendInfo = {};  % Initialize cell array for legend labels

% Loop over classes for ROC plotting
for classIndex = 0:4  % Assuming 5 classes indexed from 0 to 4
    % Creating binary labels for current class
    binaryLabelsKNN = Knnlabels == classIndex;
    binaryLabelsCNN = Cnnlabels == classIndex;

    % Calculate and Plot ROC for KNN
    if sum(binaryLabelsKNN) > 0
        [Xknn, Yknn, ~, AUCvalue] = perfcurve(double(binaryLabelsKNN), scoresKNN(2:end, classIndex+1), true);
        hKNN = plot(Xknn, Yknn, '-', 'LineWidth', 2, 'Color', colors(classIndex + 1, :)); % Plot KNN ROC Curve
        legendInfo{end+1} = sprintf('Class %d KNN', classIndex);
        hPlots(end+1) = hKNN;
        AUCknn(1,classIndex+1) = AUCvalue;
    end
   
    % Calculate and Plot ROC for CNN
    if sum(binaryLabelsCNN) > 0
        [Xcnn, Ycnn, ~, AUCvalue] = perfcurve(double(binaryLabelsCNN), scoresCNN(:, classIndex+1), true);
        hCNN = plot(Xcnn, Ycnn, '--', 'LineWidth', 2, 'Color', colors(classIndex + 1, :)); % Plotting CNN ROC Curve
        legendInfo{end+1} = sprintf('Class %d CNN', classIndex);
        hPlots(end+1) = hCNN;
        AUCcnn(1, classIndex+1) = AUCvalue;
    end
end

% Adding a reference line for no-skill classifier
hRef = plot([0, 1], [0, 1], 'k--'); 
legendInfo{end+1} = 'No-skill Line';
hPlots(end+1) = hRef;

% Add legend
legend(hPlots, legendInfo, 'Location', 'Best');

% Labels, title and grid
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('One-vs-All ROC Curves for KNN and CNN Models');

% Release the plot hold
hold off;

% Print AUC values for each class
disp('AUC values for KNN by class:');
disp(AUCknn);
disp('AUC values for CNN by class:');
disp(AUCcnn);