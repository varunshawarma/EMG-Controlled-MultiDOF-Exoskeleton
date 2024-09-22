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

% KNN Decision Boundaries Visualization
Idxs = selectBestChannels(states, features, 2);
features = features(Idxs,:);
data = [states' features'];
% Number of data points
nData = size(data, 1);

% Create a partition object for a random 70-30 split
c = cvpartition(nData, 'Holdout', 0.3);

% Indices for training and testing sets
trainIdx = training(c);
testIdx = test(c);

% Split the data
X_train = data(trainIdx, 2:end);  % Features for training
Y_train = data(trainIdx, 1);      % Labels for training
X_test = data(testIdx, 2:end);    % Features for testing
Y_test = data(testIdx, 1);        % Labels for testing

% Train kNN model
k = 2;  % Number of neighbors

reducedData = data(:, 2:end);  % Keeping only the first two principal components

% Split the reduced data according to your cvpartition object
X_train_reduced = reducedData(trainIdx, :);
X_test_reduced = reducedData(testIdx, :);

% Train kNN model on the reduced training data
knnModel = fitcknn(X_train_reduced, Y_train, 'NumNeighbors', k, 'Distance', 'euclidean', 'DistanceWeight', 'inverse');

% Create a grid for plotting decision boundaries
feature1Range = linspace(min(reducedData(:,1)), max(reducedData(:,1)), 300);
feature2Range = linspace(min(reducedData(:,2)), max(reducedData(:,2)), 300);
[x1Grid, x2Grid] = meshgrid(feature1Range, feature2Range);
xGrid = [x1Grid(:), x2Grid(:)];  % Grid for prediction

% Predict over the grid
predLabels = predict(knnModel, xGrid);

% Reshape the predicted labels to be a matrix for contourf
Z = reshape(predLabels, size(x1Grid));

% Visualize the decision boundaries with filled contours
fig = figure;
set(fig, 'renderer', 'painters');
hold on;
contourf(x1Grid, x2Grid, Z, unique(predLabels), 'LineStyle', 'none');  % Fill regions

% Define a new, more distinct colormap
newColormap = [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8; 0.9 0.5 0; 1 0 1];
colormap(newColormap);  % Apply the new colormap

% Overlay the testing points
% scatter(X_test_reduced(:,1), X_test_reduced(:,2), 50, Y_test, 'filled', 'MarkerEdgeColor', 'k');

scatter(X_train_reduced(:,1), X_train_reduced(:,2), 50, Y_train, 'filled', 'MarkerEdgeColor', 'k');

% Create dummy objects for legend entries corresponding to the contour regions
hold on;
for i = 1:length(unique(predLabels))
    h(i) = patch(NaN, NaN, newColormap(i,:), 'LineWidth', 2);
end

% Labels and legend
xlabel('Channel 1');
ylabel('Channel 2');
title('kNN Classification with Decision Boundaries');
legend(h, {'State 0', 'State 1', 'State 2', 'State 3', 'State 4'}, 'Location', 'best');
hold off;