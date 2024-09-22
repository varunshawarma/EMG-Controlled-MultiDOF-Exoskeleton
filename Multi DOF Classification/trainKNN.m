function [knnModel, accuracy, score] = trainKNN(Kinematics, Features)

    [Kinematics, Features, varargout] = alignTrainingData_jag(Kinematics, Features, 1:192, 'standard');
    
    % Defining States
    window = 10;
    state = zeros(1, size(Kinematics,2));
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

    states = state;
    features = Features;

    data = [states' features'];

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
    [Y_pred, score] = predict(knnModel, X_test);
    
    accuracy = sum(Y_pred == Y_test) / numel(Y_test);
    fprintf('Accuracy of the kNN model is: %.2f%%\n', accuracy * 100);
    
    % Perform k-fold cross-validation across the entire dataset
    kFold = 10;
    cvModel = fitcknn(data(:, 2:end), data(:, 1), 'NumNeighbors', k, 'DistanceWeight', 'inverse', 'CrossVal', 'on', 'KFold', kFold);
    
    % Assess the model's cross-validated performance
    cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'ClassifError');
    fprintf('Cross-validated Accuracy is: %.2f%%\n', cvAccuracy * 100);

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