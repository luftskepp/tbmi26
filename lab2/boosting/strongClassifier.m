function [H] = strongClassifier(X,theta,alpha)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
H = zeros(1,size(X,2));
H = sign(alpha'*weakClassifier(X,theta));
end

