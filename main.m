% =========================================================================
%             MPA-ABS 2025 PROJECT
% =========================================================================
%   INPUT DATA
%   -folder contains following files:
%       -dataset.mat            file with dataset and target
%       -main.m                 main script for data evaluation
%       -F1skore.m              function for results
%       -apneaDetection.m       YOUR FUNCTION FOR EVALUATION
% 
% DATA DESCTIPTION
% Dataset contains data from PSG study, only ECG and PPG recordings are
% left for classification. Additionally dataset contains 3 annotation
% "signals" Central, Obstruct and Hypo, which have value of 1 during the 
% event of corresponding apnea type, if any apnea was annotated by expert.
%
%  Type field can be either 
%  0 - normal sleep
%  1 - sleep apnea or hypopnea was detected
%
% PLEASE DO NOT MODIFY THIS FILE, all of your custom code should be in 
% apneaDetection.m file (or other supporting files, called by this
% function).
% ======================================================================
%% data load 
clearvars
close all hidden
clc

showresults = 1; % 0 or 1 whether you want to write results into command window
load("dataset_val.mat")
Target = [dataset.Type].';

N=(length(Target));
Result=zeros(N,1);

%% main loop for evaluation
for i=1:N
    data=dataset(i);
    
    %===================================================%
    %--------------YOUR FUNCTION IS HERE----------------%
    class = apneaDetection(data);
    %===================================================%
    Result(i) = class;
    
    if showresults
        if class == Target(i)
            disp(['Record number ' num2str(i) ' is CORRECT (' num2str(class) ')'])
        else
            disp(['Record number ' num2str(i) ' is classified as ' num2str(class) ' and should be ' num2str(Target(i))])
        end
    end

end
%% Results calculation
SCORE = F1skore(Target,Result)    % calculates Score
%% Evaluation function
function SCORE = F1skore(Target, Result)

% Confusiom matrix for 2 classes (0 a 1)
confusionMat = zeros(2,2);

for i = 1:length(Target)
    if Target(i) == 0 && Result(i) == 0
        confusionMat(1,1) = confusionMat(1,1) + 1; % TN
    elseif Target(i) == 0 && Result(i) == 1
        confusionMat(1,2) = confusionMat(1,2) + 1; % FP
    elseif Target(i) == 1 && Result(i) == 0
        confusionMat(2,1) = confusionMat(2,1) + 1; % FN
    elseif Target(i) == 1 && Result(i) == 1
        confusionMat(2,2) = confusionMat(2,2) + 1; % TP
    end
end

TN = confusionMat(1,1);
FP = confusionMat(1,2);
FN = confusionMat(2,1);
TP = confusionMat(2,2);

% Precision, Recall, F1
if TP+FP == 0
    Precision = 0;
else
    Precision = TP / (TP + FP);
end

if TP+FN == 0
    Recall = 0;
else
    Recall = TP / (TP + FN);
end

if Precision+Recall == 0
    SCORE = 0;
else
    SCORE = 2 * (Precision * Recall) / (Precision + Recall);
end

% Visual
f = figure('Units','pixels','Position',[0 0 800 600]);
confusionchart(confusionMat, {'0','1'});

end