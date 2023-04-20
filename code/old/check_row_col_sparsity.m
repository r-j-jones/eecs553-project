function [pctRowsOk,pctColsOk] = check_row_col_sparsity(S,r,alpha,makeFig)

% [pctRowsOk,pctColsOk] = check_row_col_sparsity(S,r,alpha[,makeFig])
%
% Compute row and column sparsity levels for given alpha, and see if how
% many rows/cols fall within theoretical limit in assumption (A4)
%
% (A4): S[d x N] has at most (n-r)/[2(r+1)^alpha] outliers per row and
%         (d-r)/[2(r+1)^(alpha-1)] outliers per col, with alpha>=1

if nargin<4
    makeFig=false;
end

[i,j,~]=find(S);

[nrows,ncols] = size(S);

rowOutliers = zeros(size(S,1),1);
for ii=1:nrows
    rowOutliers(ii) = nnz(i==ii);
end

colOutliers = zeros(size(S,2),1);
for jj=1:ncols
    colOutliers(jj) = nnz(j==jj);
end

rowSpThr = (ncols-r)/((2*(r+1))^alpha);
colSpThr = (nrows-r)/((2*(r+1))^(alpha-1));

pctRowsOk = nnz(rowOutliers<=rowSpThr)/nrows;
pctColsOk = nnz(colOutliers<=colSpThr)/ncols;

if makeFig
    figure;

    subplot(121);
    hold on;
    plot(rowOutliers);
    plot(1:nrows,repmat(rowSpThr,nrows,1));
    legend('Actual # outliers','Theor. thresh.');
    xlabel('Row #');
    ylabel('Outliers/row')
    title(['Row sparsity (alpha=' num2str(alpha) ')']);
    
    subplot(122);
    hold on;
    plot(colOutliers);
    plot(1:ncols,repmat(colSpThr,ncols,1));
    legend('Actual # outliers','Theor. thresh.');
    xlabel('Col #');
    ylabel('Outliers/col')
    title(['Col sparsity (alpha=' num2str(alpha) ')']);

    drawnow;
end

end
