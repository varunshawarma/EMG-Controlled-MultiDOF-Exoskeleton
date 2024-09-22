clear;clc;close all;
caleb_file = "\\neurorobotsrv\D\FeedbackDecode\20240711-142221\TrainingData_20240711-142221_142544.kdf";
addpath(genpath("../../../Students/Grads/CDO/CustomMatlabFunctions"));

[Kinematics,Features,~,~,NIPTime] = readKDF("\\neurorobotsrv\D\SmartHome\PvNP_Wrist_Forearm\S1_P\TaskData_20230308-174844.kdf");
% [Kinematics,Features,~,~,NIPTime] = readKDF(caleb_file);

%%
figure();
hold on;
for i = 1:height(Kinematics)
    plot(NIPTime,Kinematics(i,:)-2*i);
end

logicalArray_Grasp_Open = Kinematics(1,:) & Kinematics(2,:) & Kinematics(3,:) & Kinematics(4,:) & Kinematics(5,:);
logicalArray_Supinate_Pronate = logical(Kinematics(12,:));

logicalArray = logicalArray_Supinate_Pronate | logicalArray_Grasp_Open;

plot(NIPTime, logicalArray,'LineWidth',1.5, 'Color', "#000000");


% window = 100;
% for i = 1:length(logicalArray)-window
%     if logicalArray(i) == 1 && logicalArray(i+window) == 1
%         logicalArray(i:i+window) = 1;
%     end
% end
% 
% plot(NIPTime, logicalArray,'LineWidth',1.5, 'Color', "#ff0000");
% plot(NIPTime, Features);


%% trim the data

NIPTime = NIPTime(logicalArray);
Kinematics = Kinematics(:,logicalArray);
Features = Features(:,logicalArray);

