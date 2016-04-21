%% Lab2

% Load face and non-face data and plot a few examples
load faces, load nonfaces
faces = double(faces); nonfaces = double(nonfaces);
figure(1)
colormap gray
for k=1:25
subplot(5,5,k), imagesc(faces(:,:,10*k)), axis image, axis off
end
figure(2)
colormap gray
for k=1:25
subplot(5,5,k), imagesc(nonfaces(:,:,10*k)), axis image, axis off
end

%% generate haar feats
nbrHaarFeatures = 100;
nbrTrainExamples = 500;
% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
figure(3)
colormap gray
for k = 1:min(25,nbrHaarFeatures)
subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2])
axis image,axis off
end

% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

%% train weak classifiers



% set number of classifiers and initialize weights
numWeakClassifiers = 10; 
di = 1/size(xTrain,2)*ones(1,size(xTrain,2)); 
alpha = zeros(numWeakClassifiers,1);

theta = zeros(size(xTrain,1)+1,numWeakClassifiers);
% loop for all classifiers h(x;theta) = theta'*x
% ii - classifier ii    (h_ii)
% jj - feature jj       (x_ii)
% kk - sample kk        (X^jj)
for ii = 1:numWeakClassifiers
    % init error and polarity for classifier
    epsilonmin = Inf;
    ptmp = 1;
    for jj = 1:size(xTrain,1) % loop for all features
        for kk = 1:size(xTrain,2) % loop for all samples
            % calc error
            etmp = sum(di.*(yTrain ~= sign(ptmp*(xTrain(jj,:)-xTrain(jj,kk)))));
            % if larger than 0.5, switch polarity
            if etmp > .5
                etmp = 1-etmp;
                ptmp = -ptmp;
            end
            % if smaller than last error, update theta for classifier
            if etmp < epsilonmin
                epsilonmin = etmp;
                % reset theta and update
                theta(1,ii) = -xTrain(jj,kk);
                theta(2:end,ii) = 0;
                theta(jj+1,ii) = 1;
                % include polarity
                theta(:,ii) = ptmp*theta(:,ii);
            end
        end
    end
    % update weights
    alpha(ii) = 1/2*log((1-epsilonmin)/epsilonmin);
    di = di.*exp(-alpha(ii)*yTrain.*weakClassifier(xTrain,theta(:,ii)));
    di = di./sum(di);
end

%% Strong classifier
% test on train images
Htrain = strongClassifier(xTrain,theta,alpha);

%% test on other images
nbrTestExamples = 1500;
testImages = cat(3,faces(:,:,nbrTrainExamples+1:nbrTrainExamples+nbrTestExamples),...
    nonfaces(:,:,nbrTrainExamples+1:nbrTrainExamples+nbrTestExamples));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

Htest = strongClassifier(xTest,theta,alpha);

correctClassifiedImages = sum(Htest == yTest)
ratio = correctClassifiedImages / (nbrTestExamples*2)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot for number of classifiers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbrHaarFeatures = 50;
nbrTrainExamples = 200;

haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];
%%
numWeakClassifiers = 30; 
accuracyTrain = zeros(1,numWeakClassifiers);
thetaTrain = cell(numWeakClassifiers);
alphatrain = cell(numWeakClassifiers);
for numWeakCl = 1:numWeakClassifiers
    di = 1/size(xTrain,2)*ones(1,size(xTrain,2));
    alpha = zeros(numWeakCl,1);
    
    theta = zeros(size(xTrain,1)+1,numWeakCl);
    % loop for all classifiers h(x;theta) = theta'*x
    % ii - classifier ii    (h_ii)
    % jj - feature jj       (x_ii)
    % kk - sample kk        (X^jj)
    for ii = 1:numWeakCl
        % init error and polarity for classifier
        epsilonmin = Inf;
        ptmp = 1;
        for jj = 1:size(xTrain,1) % loop for all features
            for kk = 1:size(xTrain,2) % loop for all samples
                % calc error
                etmp = sum(di.*(yTrain ~= sign(ptmp*(xTrain(jj,:)-xTrain(jj,kk)))));
                % if larger than 0.5, switch polarity
                if etmp > .5
                    etmp = 1-etmp;
                    ptmp = -ptmp;
                end
                % if smaller than last error, update theta for classifier
                if etmp < epsilonmin
                    epsilonmin = etmp;
                    % reset theta and update
                    theta(1,ii) = -xTrain(jj,kk);
                    theta(2:end,ii) = 0;
                    theta(jj+1,ii) = 1;
                    % include polarity
                    theta(:,ii) = ptmp*theta(:,ii);
                end
            end
        end
        % update weights
        alpha(ii) = 1/2*log((1-epsilonmin)/epsilonmin);
        di = di.*exp(-alpha(ii)*yTrain.*weakClassifier(xTrain,theta(:,ii)));
        di = di./sum(di);
    end
    Htrain = strongClassifier(xTrain,theta,alpha);
    accuracyTrain(numWeakCl) = sum(Htrain==yTrain)/length(Htrain);
    thetaTrain{numWeakCl} = theta;
    alphaTrain{numWeakCl} = alpha;
end

figure(1)
plot(accuracyTrain); xlabel 'number of weak classifiers'; ylabel 'accuracy'; 
title 'Accuracy for training data';

%% testing classifier

nbrTestExamples = 1500;
testImages = cat(3,faces(:,:,nbrTrainExamples+1:nbrTrainExamples+nbrTestExamples),...
    nonfaces(:,:,nbrTrainExamples+1:nbrTrainExamples+nbrTestExamples));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

accuracyTest = zeros(1,numWeakClassifiers);
for numWeakCl = 1:numWeakClassifiers
    Htest = strongClassifier(xTest,thetaTrain{numWeakCl},alphaTrain{numWeakCl});
    accuracyTest(numWeakCl) = sum(Htest == yTest)/length(Htest);
end
figure(2)
plot(accuracyTest); xlabel 'number of weak classifiers'; ylabel 'accuracy'; 
title 'Accuracy for test data';
