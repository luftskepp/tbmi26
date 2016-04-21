%%%%%%%%%%
% LAB 3
%%%%%%%%%
load 'countrydata';

% normalize data
datanorm = (countrydata - repmat(min(countrydata,[],2),1,size(countrydata,2)))...
    ./repmat(max(countrydata,[],2)-min(countrydata,[],2),1,size(countrydata,2));
%% PCA
% covariance matrix
datacov = (datanorm - repmat(mean(datanorm,2),1,size(datanorm,2)));
datacov = datacov*datacov'/size(datanorm,2);
figure(1); imagesc(datacov); colorbar; title 'covariance';

% correlation matrix
varmatrix = repmat(var(datanorm,1,2),1,size(datanorm,1));
datacorr = datacov./sqrt(varmatrix)./sqrt(varmatrix');
figure(2); imagesc(datacorr); colorbar; title correlation;



% extract eigenvalues and vectors
[E,D]= sorteig(datacorr);
figure(3); stem(D); title 'eigenvalues C';

% plot PC
figure(4); clf; title 'PCA'; hold on;
markers = {'go', 'bo','mo'}; 
for cclass = 1:length(unique(countryclass))
plot(E(:,1)'*datanorm(:,countryclass == cclass-1),E(:,2)'*datanorm(:,countryclass == cclass-1),markers{cclass});
end;
plot(E(:,1)'*datanorm(:,41),E(:,2)'*datanorm(:,41),'rx');
xlabel 'e1'; ylabel 'e2'; legend 'developing' 'inbetween' 'industrialized' 'Georgia';
hold off;

%% FLD
%countryclass(countryclass==0) = 1;
Ctot = zeros(size(datanorm,1));
for cclass = 0:2:2
    tmpdata = datanorm(:,countryclass==cclass);
    tmpcov = (tmpdata - repmat(mean(tmpdata,2),1,size(tmpdata,2)));
    Ctot = Ctot + tmpcov*tmpcov'/size(tmpdata,2);
end;

w = Ctot\(mean(datanorm(:,countryclass==0),2)-mean(datanorm(:,countryclass==2),2));

FLD1 = (w'*datanorm(:,countryclass==0));
FLD2 = (w'*datanorm(:,countryclass==2));


figure(5); clf; 
scatter(FLD1,zeros(size(FLD1)));hold on; scatter(FLD2,zeros(size(FLD2)));
