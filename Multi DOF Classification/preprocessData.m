function [states, features, Idxs] = preprocessData(file_path)
% Inputs:
%   file_path - String specifying the path to the KDF file.
%
% Outputs:
%   states    - 1xN vector of state labels corresponding to each data point.
%               States include:
%               0: Neutral
%               1: Grasp
%               2: Open
%               3: Pronation
%               4: Supination
%
%   features  - MxN matrix of selected EMG features/channels after preprocessing
%               M is the number of selected channels (default 48)
%               N is the number of data points
%
%   Idxs      - 1xM vector of indices of the selected EMG channels

    addpath(genpath("../../../Students/Grads/CDO/CustomMatlabFunctions"))
    %addpath("\\Neurorobotrt1\c\Users\Administrator\Box\NeuroRoboticsLab\JAGLAB\Projects\Adaptive EMG control")
    
    [Kinematics, Features,~,~,NIPTime] = readKDF(file_path);

    [Kinematics, Features, varargout] = alignTrainingData_jag(Kinematics, Features, 1:192, 'standard');

    % Filter relevant data points
    logicalArray_Grasp_Open = Kinematics(1,:) & Kinematics(2,:) & Kinematics(3,:) & Kinematics(4,:) & Kinematics(5,:);
    logicalArray_Supinate_Pronate = logical(Kinematics(12,:));
  
    logicalArray = logicalArray_Supinate_Pronate | logicalArray_Grasp_Open;
    window = 100;
    for i = 1:length(logicalArray)-window
        if logicalArray(i) == 1 && logicalArray(i+window) == 1
            logicalArray(i:i+window) = 1;
        end
    end
    
    NIPTime = NIPTime(logicalArray);
    Kinematics = Kinematics(:,logicalArray);
    Features = Features(:,logicalArray);

    % Defining States 
    state = zeros(1, length(NIPTime));
    count = 1;
    window = 10;
    while count <= length(NIPTime)
        state(count) = findState(count, Kinematics);
    
        % Window
        windowcount = min(count+window,length(NIPTime));
        if findState(windowcount, Kinematics) == state(count)
            state(count:windowcount) = state(count);
            count = windowcount+1;
        else
            count = count+1;
        end
    end

    %tabulate(state);
    
    % Feature/Channel Selection
    states = state;
    numIdxs = 48;
    Idxs = selectBestChannels(states, Features, numIdxs);
    features = Features(Idxs,:);

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
        else
            s = 0; % Neutral
        end
    end
end