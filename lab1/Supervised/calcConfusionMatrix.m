function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

for ii=1:numClasses
    guessInd = (Lclass == ii);
    for jj = 1:numClasses
       trueInd = (Ltrue == jj);
       cM(jj,ii) = sum(guessInd.*trueInd);
   end
end

end

