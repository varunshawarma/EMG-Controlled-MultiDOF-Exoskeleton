function [predKinematics] = testKNN(model, features, varargin)

X = features';
[Y,score] = predict(model, X);

% % Confidence Level Filter
% threshold = 0.85;
% confidence = max(score, [], 2);
% if confidence < threshold
%     Y = varargin;
% end

predKinematics = zeros(12, length(features));
predKinematics(:, 1) = 0;  % Initialize first column with first kinematic data

for i = 2:length(features)
   prediction = Y(i);
    switch prediction
    case 1
        predKinematics(1:5,i) = 1;
    case 2
        predKinematics(1:5,i) = -1;
    case 3
        predKinematics(6,i) =1;  
    case 4
        predKinematics(6,i) = -1; 
    otherwise
        predKinematics(:,i) = predKinematics(:,i-1);  % Carry forward if no change
    end
end

end




