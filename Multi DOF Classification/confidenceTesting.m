clear; clc; close all;

% Sensitive paths replaced with placeholders
addpath(genpath("path_to_custom_functions"));
addpath("path_to_haptix_offline");
addpath("path_to_project_files");
addpath("path_to_multi_dof_classification");

% Placeholders for sensitive data file paths
files = {"path_to_data/S1_P/TaskData_1.kdf", 
         "path_to_data/S1_NP/TaskData_2.kdf",
         "path_to_data/S2_P/TaskData_3.kdf", 
         "path_to_data/S2_NP/TaskData_4.kdf", 
         "path_to_data/S3_P/TaskData_5.kdf", 
         "path_to_data/S3_NP/TaskData_6.kdf"};

file_path = files{1};

[Kinematics, Features,~,~,NIPTime] = readKDF(file_path);
[states, features] = preprocessData(file_path);

%% Low Confidence Prediction Visualisation
[knnModel, overallAccuracy, cvAccuracy] = knnClassifier(states, features);
[Preds,score] = predict(knnModel, features');
Y_pred = Preds;
confidence = max(score, [], 2);  % Get the maximum score for each prediction as confidence

figure;
hold on;
histogram(confidence, 10);  % Adjust the number of bins as necessary
title('Histogram of Prediction Confidence');
xlabel('Confidence Score');
ylabel('Frequency');

lowConfidenceThreshold = 0.85;  % Define low confidence threshold
lowConfIndices = find(confidence < lowConfidenceThreshold);  % Indices of low confidence predictions

time = 1:length(Preds); % Time vector

figure;
% Plotting actual states using a line plot with markers
plot(time, states', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 5, 'Color', 'b', 'DisplayName', 'Actual States');

% Plotting predictions using a bar plot for contrast
bar(time, Y_pred, 'FaceColor', 'r', 'DisplayName', 'Predictions'); % Red color

% Color changed for low confidence bars
bar(time(lowConfIndices), Preds(lowConfIndices), 'FaceColor', 'k', 'DisplayName', 'Low Confidence Predictions');  % BLACK

% Adding labels, legend, and title for clarity
xlabel('Time Index');
ylabel('State');
title('Comparison of Predictions and Actual States');
legend('show'); % Show legend to identify the plots
hold off;

% Normalized confusion matrix
confusionchart(states', Y_pred, 'Normalization', 'row-normalized');
title('Normalized Confusion Matrix for Validation Data');

%% Confidence Level Based Control - Evaluation
[knnModel, overallAccuracy, cvAccuracy, score] = knnClassifier(states, features);
[Y_pred, score] = predict(knnModel, features');
confidence = max(score, [], 2);  % Get the maximum score for each prediction as confidence
lowConfidenceThreshold = 0.85;  % Define low confidence threshold
lowConfIndices = find(confidence < lowConfidenceThreshold);  % Indices of low confidence predictions

lowConfFeatures = features(:,lowConfIndices);  % Features for low confidence predictions
lowConfPredictions = Y_pred(lowConfIndices);  % Predicted classes of low confidence predictions
trueLabelsLowConf = states(lowConfIndices);

modifiedPredictions = Y_pred;

for i = 1:length(lowConfIndices)
    idx = lowConfIndices(i);
    if idx > 1
        modifiedPredictions(idx) = modifiedPredictions(idx - 1);
    end
end

confusionchart(trueLabelsLowConf, lowConfPredictions, 'Normalization', 'row-normalized');
title('Normalized Confusion Matrix for Low Confidence Predictions');

% confusionchart(trueLabelsLowConf, modifiedPredictions(lowConfIndices), 'Normalization', 'row-normalized');
% title('Normalized Confusion Matrix for Modified Low Confidence Predictions');
