%% kNN optimal k using cross-validation

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr =4; % Change this to load new data 
kmin = 91;
kmax = 99;
clc;
[accM kopt] = optKfromCrossValid(3,dataSetNr,kmax,kmin)

%%

