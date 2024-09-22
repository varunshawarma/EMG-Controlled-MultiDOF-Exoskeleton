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

clc;
handMotion = zeros(1,length(features));
wristMotion = zeros(1,length(features));
handMotion(1,1) = 700;
wristMotion(1,1) = 700;

delta = 1;
for x = 2:length(features)
    switch state(1,x)
        case 1
            handMotion(1,x) = min(handMotion(1,x-1)+delta,1400);
            wristMotion(1,x) = wristMotion(1,x-1);
        case 2
            handMotion(1,x) = max(handMotion(1,x-1)-delta,0);
            wristMotion(1,x) = wristMotion(1,x-1);
        case 3
            wristMotion(1,x) = min(wristMotion(1,x-1)+delta,1400);
            handMotion(1,x) = handMotion(1,x-1);
        case 4 
            wristMotion(1,x) = max(wristMotion(1,x-1)-delta,0);
            handMotion(1,x) = handMotion(1,x-1);
        case 0
            wristMotion(1,x) = wristMotion(1,x-1);
            handMotion(1,x) = handMotion(1,x-1);
    end
end

% Plotting Features, States and Kinematics vs Time
fig = figure;  
set(fig, 'renderer', 'painters');
hold on; 

hFeatures = plot(1:length(features), features);

hStates = plot(1:length(states), states*300, 'k-', 'LineWidth', 2, 'DisplayName', 'States (scaled)');

hHandMotion = plot(1:length(handMotion), handMotion, 'r-', 'LineWidth', 2, 'DisplayName', 'Hand Motion');

hWristMotion = plot(1:length(wristMotion), wristMotion, 'b-', 'LineWidth', 2, 'DisplayName', 'Wrist Motion');

xlabel('Time');
ylabel('Feature Value');
title('Features Across Time');

legend([hStates, hHandMotion, hWristMotion], 'States', 'Hand Motion', 'Wrist Motion');

hold off;    
