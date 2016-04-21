function [kAcc,kOpt] = optKfromCrossValid(n,dataSetNr,kmax,kmin)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Select a subset of the training features
if nargin<4
    kmin =1;
end;

[X, D, L] = loadDataSet( dataSetNr ); 
numBins = n; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = Inf; % Number of samples per label per bin, set
                                % to inf for max number (total number is
                                % numLabels*numSamplesPerBin)
selectAtRandom = false; % true = select features at random, 
                       % false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};
% Use kNN to classify data
% Note: you have to modify the kNN() function yourselfs.

% Set the number of neighbors
accM = zeros(size(kmin:kmax));

for ii = kmin:kmax
    k = ii;
    LkNN = kNN(Xt{1}, k, [Xt{2:end}], [Lt{2:end}]);
    cM = calcConfusionMatrix( LkNN, Lt{1});
    % The accuracy
    acc = calcAccuracy(cM);
    accM(ii-kmin+1) = accM(ii-kmin+1)+acc;
    
    for jj = 2:n-1
        LkNN = kNN(Xt{jj},k,[Xt{2:jj-1} Xt{jj+1:end}], [Lt{2:jj-1} Lt{jj+1:end}]);
        % The confucionMatrix
        cM = calcConfusionMatrix( LkNN, Lt{jj});
        acc = calcAccuracy(cM);
        accM(ii-kmin+1) = accM(ii-kmin+1)+acc;
    end;
        
    LkNN = kNN(Xt{end}, k, [Xt{1:end-1}], [Lt{1:end-1}]);
    cM = calcConfusionMatrix( LkNN, Lt{end});
    % The accuracy
    acc = calcAccuracy(cM);
    accM(ii-kmin+1) = accM(ii-kmin+1)+acc;
    
end

accM = accM./n;
[~, kOpt] = max(accM);
kOpt = kOpt+kmin-1;
kAcc = accM;
end

