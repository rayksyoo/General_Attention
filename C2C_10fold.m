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

%% CPM+C2C modeling
nPerm =  size(permList,1);
meanPCA = 0;    nPCA = 50;    nPLS = 10;    thr = 0.05;

for np =1:nPerm
    for nf = 1:nFold
        fprintf('\t%d/%d fold %d/%d iteration \n',  nf, nFold, np, nPerm)
        subjTest = permList(np,[subjKlist(nf,1):subjKlist(nf,2)]);
        subjTrain = setdiff([1:size(behav_all,1)],subjTest);
        for t1 = 4:5
            for t2 = 1:3
                matPred{t1-3}(:,subjTest,t2) = c2c_2sets(mat_all(:,subjTrain, t1)', mat_all(:,subjTrain, t2)', mat_all(:,subjTest, t1)', nPCA, nPLS, meanPCA)';
                [~, CPMpredRbase(subjTest, t2,t1-3)] = cpm_lr_D1D2_(mat_all(:,subjTrain, t1), matPred{t1-3}(:,subjTest,t2), behav_all(subjTrain,t2), behav_all(subjTest,t2), thr);
                [~, CPMpredTbase(subjTest, t2,t1-3)] = cpm_lr_D1D2_(mat_all(:,subjTrain, t2), matPred{t1-3}(:,subjTest,t2), behav_all(subjTrain,t2), behav_all(subjTest,t2), thr);
            end
        end
    end

    % save matPred* and CPMpred* files
end



