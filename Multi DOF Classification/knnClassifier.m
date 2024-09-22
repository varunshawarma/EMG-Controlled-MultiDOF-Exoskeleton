function [knnModel, accuracy, cvAccuracy, score, predictT, Y_pred, Y_test] = knnClassifier(states, features)
% Inputs:
%   states   - 1xN vector of state labels corresponding to each sample.
%   features - MxN matrix of features, where M is the number of features (channels)
%              and N is the number of samples.
%
% Outputs:
%   knnModel  - Trained k-Nearest Neighbors (kNN) model.
%   accuracy  - Scalar value representing the classification accuracy of the kNN on the test set.
%   cvAccuracy- Scalar value representing the cross-validated accuracy of the kNN model.
%   score     - NxC matrix of prediction scores, where N is the number of test samples and C is the number of prediction classes.
%   predictT  - 1x500 vector of prediction times for 500 test samples.
%   Y_pred    - Nx1 vector of predicted class labels by the kNN for the test set.
%   Y_test    - Nx1 vector of true class labels for the test set.


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
knnModel = fitcknn(X_train, Y_train, 'NumNeighbors', k, 'DistanceWeight', 'inverse');

predictT = zeros(1,500);
for i = 1:500
    tic;
    singleP = predict(knnModel, X_test(i, :));  % Predicting only the first sample
    predictT(1,i) = toc;
end

[Y_pred, score] = predict(knnModel, X_test);


accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf('Accuracy of the kNN model is: %.2f%%\n', accuracy * 100);

% Perform k-fold cross-validation across the entire dataset
kFold = 10;
cvModel = fitcknn(data(:, 2:end), data(:, 1), 'NumNeighbors', k, 'DistanceWeight', 'inverse', 'CrossVal', 'on', 'KFold', kFold);

% Assess the model's cross-validated performance
cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'ClassifError');
fprintf('Cross-validated Accuracy is: %.2f%%\n', cvAccuracy * 100);

% figure; 
% % confusionchart(Y_test, Y_pred);
% % title('Confusion Matrix for Validation Data');
% 
% % Normalized confusion matrix
% confusionchart(Y_test, Y_pred, 'Normalization', 'row-normalized');
% title('Normalized Confusion Matrix for Validation Data');


end
