%% Variables

% behav_all:  z-scored behavior performances of individuals (in rows) and three tasks (in columns)
% mat_all:     vectorized connectomes of individuals for five states in 3-dim (edge x subject x states)
%                 five states: task 1, task 2, task 3, rest, and movie
% nFold:       10 for 10-fold.
% thr:           0.05 for a feature selction threshold in CPM modeling
% permList:   1000 iterations x shuffled subjects in 2-dim (iteration x shuffling label)
% subjKlist:   10 x 2 matrix, defining the first and last subjects number in 10 folds. (fold x subject number)
% subjKnum: 10 x 1 matrix, defining the number of subjects in each fold (should agree with subjKlist) 

% CPMpred*:   individuals' predicted scores (1st dim) with training data (2nd dim) and testing data (3rd dim) 

%% Universal attention connectome lookup table
for e = 1:size(mat_all,1)
    [~, temp_idx] = max(abs(mean(squeeze(mat_all(e,:,1:3)))));
    aa5(e,1) = temp_idx;
    mat_task(e,:) =mat_all(e,:,temp_idx);    clear temp_idx
end

%% Universal attention modeling
nPerm =  size(permList,1);
meanPCA = 0;    nPCA = 50;    nPLS = 10;    thr = 0.05;

for np = 1:nPerm
    for nf = 1:nFold
        fprintf('\t%d/%d fold %d/%d iteration \n',  nf, nFold, np, nPerm)
        subjTest = permList(np,[subjKlist(nf,1):subjKlist(nf,2)]);
        subjTrain = setdiff([1:size(behav_all,1)],subjTest);
        
        matPred(:,subjTest) = c2c_2sets(mat_all(:,subjTrain, 4)', mat_task(:,subjTrain)', mat_all(:,subjTest, 4)', nCompPCA, nCompPLS, meanPCA)';
        [~, CPMpred_universal(subjTest,1)] = cpm_lr_D1D2_union(mat_task(:,subjTrain), matPred(:,subjTest), mean(behav_all(subjTrain, :), 2), behav_all(subjTest, 1), thr, behav_all(subjTrain, :) );
    end

    % performance: correlation
    [temp_R temp_P] = corr(behav_all, squeeze(CPMpred_universal));
    predCorr_universal.R(:,:,np) = temp_R;    predCorr_universal.P(:,:,np) = temp_P;    clear temp*

    % performance: Prediction q^2
    for nf = 1:nFold
        subjTest = permList(np,[subjKlist(nf,1):subjKlist(nf,2)]);
        subjTrain = setdiff([1:size(behav_all,1)],subjTest);
        deno(subjTest,:) = behav_all(subjTest, :) - mean(behav_all(subjTrain, :));
        nume(subjTest,:) = behav_all(subjTest, :) - CPMpred_universal(subjTest);
    end
    predMSE_universal(:, np) = 1 - ( mean(nume.^2) ./mean(deno.^2));    clear deno nume        
end;    clear s t1 t2

% save matPred, CPMpred_universal, and pred*_universal files



