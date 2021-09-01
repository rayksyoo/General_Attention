function [CPMperformance, CPMpredictedScore, CPMmask, CPMmodel, CPMmaskPN, CPMperformancePN, CPMpredictedScorePN] = cpm_lr_D1D2_(conMat1, conMat2, behav1, behav2, thr, var2control, subOption)
% function [predPerformance1 predPerformance2 y_predict] = cpm_cvMain_lr_ray3(x, y, pthresh, sub_rand, subListKfold)
% CPM_LR_LOOCV    Predict individual behaviors based on the connectivity matrix constructed from the time-series
%                   with linear regression (the original methods used in Rosenberg et al. (2016) Nat Neurosci.)
% ==============================================================================================
% [ INPUTS ]
%     conMat : n2 x m,  [n2=n*(n-1)/2, where n is the number of nodes and m is the number of subjects]
%
%     behav : m x 1 (where m is the number of subjects) individual behavioral scores/performances/traits observed
%
%     thr : a threshold selecting features% -----------------------------------------------------------------------------------------------
% [ OUTPUTS ]
%     CPMperformance : CPM's prediction performance estimatied by correlating predicted score (CPMpredictedScore) and observed score (behav)
%                        Pearson's correlation coefficient r and p-value
%
%     CPMpredictedScore : predicted score
%
%     CPMmask : two sets of mask for connectivity matrix describing predicting edges
%
% Reference:
% cpm_lr_loocv.m
%
% Last update: Nov 29, 2017.
%
% Copyright 2017. Kwangsun Yoo (Yoo K), PhD
%     E-mail: kwangsun.yoo@yale.edu
% ================================================================================================

if nargin < 7;    subOption = 0;    end;
if nargin < 6;    var2control = [];    end;
if nargin < 5;    thr = 0.01;    end;

conMat1 = conMat1';    conMat2 = conMat2';

% [edgeCorr.r, edgeCorr.p] = corr(conMat1, behav1);
[edgeCorr.r, edgeCorr.p] = corr(conMat1, behav1, 'rows', 'pairwise');
CPMmask=(+(edgeCorr.r>0))-(+(edgeCorr.r<0));    CPMmask=CPMmask .* (+(edgeCorr.p<=thr));    clear edgeCorr

if ~isempty(var2control)
%     [edgeCorr.r, edgeCorr.p] = corr(conMat1, var2control);
    [edgeCorr.r, edgeCorr.p] = corr(conMat1, var2control, 'rows', 'pairwise');
%     edge2control =  1- sum(+(edgeCorr.p<=thr),2);    
    edge2use = +(sum(+(edgeCorr.p<=thr),2) == 0);    
    CPMmask = CPMmask .* edge2use;    clear edgeCorr edge2*
end

% Positive network strength and prediction
if (length(find(CPMmask>0)) > 0)
    netStr.p = nanmean(conMat1(:, CPMmask>0),2);
    CPMmodelPos = fitlm(netStr.p, behav1);
    CPMpredictedScorePN(:,1) = predict(CPMmodelPos, nanmean(conMat2(:, CPMmask>0),2));    clear CPMmodelPos;
else;     netStr.p = zeros(size(behav1));    CPMpredictedScorePN([1:size(behav2,1)],1) = nan;
end;

% Negative network strength and prediction
if (length(find(CPMmask<0)) > 0) 
    netStr.n = nanmean(conMat1(:, CPMmask<0),2);
    CPMmodelNeg = fitlm(netStr.n, behav1);
    CPMpredictedScorePN(:,2) = predict(CPMmodelNeg, nanmean(conMat2(:, CPMmask<0),2));    clear CPMmodelNeg;
else;    netStr.n = zeros(size(behav1));    CPMpredictedScorePN([1:size(behav2,1)],2)  = nan;
end;

% Summuray network strength and GLM prediction
if subOption == 0
    CPMmodel = fitlm( [netStr.p netStr.n], behav1);
    CPMpredictedScore = predict(CPMmodel, [nanmean(conMat2(:, CPMmask>0),2) nanmean(conMat2(:, CPMmask<0),2)] );    % clear CPMmodel;
elseif subOption == 1
    CPMmodel = fitlm( netStr.p-netStr.n , behav1);
    CPMpredictedScore = predict(CPMmodel, nanmean(conMat2(:, CPMmask>0),2) - nanmean(conMat2(:, CPMmask<0),2) );    % clear CPMmodel;
end

[CPMperformance.r, CPMperformance.p] = corr(CPMpredictedScore, behav2);   %, 'type', 'Spearman');
[CPMperformancePN.r, CPMperformancePN.p] = corr(CPMpredictedScorePN, behav2);   %, 'type', 'Spearman');
% [CPMperformance.r, CPMperformance.p] = corr(CPMpredictedScore, behav2, 'type', 'Spearman');

CPMmaskPN.p = CPMmask;    CPMmaskPN.p(CPMmaskPN.p<0) = 0;
CPMmaskPN.n = -CPMmask;    CPMmaskPN.n(CPMmaskPN.n<0) = 0;    
