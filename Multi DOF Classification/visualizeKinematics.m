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
