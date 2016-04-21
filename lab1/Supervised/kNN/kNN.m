function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);

for ii = 1:length(labelsOut)
    % repeat X sample features to calculate distance to all training samples Xt
    Xtmp = repmat(X(:,ii),1,size(Xt,2));
    disttmp = sqrt(sum((Xtmp-Xt).^2,1));
    
    % sort distances and pick out k shortest
    [~, distind] = sort(disttmp);
    distind = distind(1:k); % indexes of the shortest 
                            % distances to training sample
    
    NNClasses = Lt(distind);
    % compute histogram of classes among the nearest neighbors
    classHist = histc(NNClasses,1:numClasses);
    % in case of draw, remove neighbor furthest away
    while(sum(classHist==max(classHist))>1 && k>1)
        k=k-1;
        distind = distind(1:k);
        NNClasses = Lt(distind);
        classHist = histc(NNClasses,1:numClasses)
    end
    [~,sampleClass] = max(classHist);
    labelsOut(ii) = sampleClass;
end

end

