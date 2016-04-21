function [ out ] = weakClassifier(X,theta )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
X = [ones(1,size(X,2)) ; X];
out = sign(theta'*X);

end

