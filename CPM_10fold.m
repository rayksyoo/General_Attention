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

%% CPM modeling (9 models x 9 generalizations)
nPerm = size(permList,1);

for np =1:nPerm
    for nf = 1:nFold
        fprintf('\t%d/%d fold %d/%d iteration \n',  nf, nFold, np, nPerm)
        subjTest = permList(np,[subjKlist(nf,1):subjKlist(nf,2)]);
        subjTrain = setdiff([1:size(behav_all,1)],subjTest);
        for t1 = 1:3
            for t2 = 1:3
                [~, CPMpred_TT{np}(subjTest, t1, t2)] = cpm_lr_D1D2_(mat_all(:,subjTrain, t1), mat_all(:,subjTest, t2), behav_all(subjTrain, t1), behav_all(subjTest, t2), thr);
                [~, CPMpred_RT(subjTest, t1, t2)] = cpm_lr_D1D2_(mat_all(:,subjTrain, 4), mat_all(:,subjTest, t2), behav_all(subjTrain, t1), behav_all(subjTest, t2), thr);
                [~, CPMpred_MT(subjTest, t1, t2)] = cpm_lr_D1D2_(mat_all(:,subjTrain, 5), mat_all(:,subjTest, t2), behav_all(subjTrain, t1), behav_all(subjTest, t2), thr);
            end
            
            [~, CPMpred_TR(subjTest, t1)] = cpm_lr_D1D2_(mat_all(:,subjTrain, t1), mat_all(:,subjTest, 4), behav_all(subjTrain, t1), behav_all(subjTest, t1), thr);
            [~, CPMpred_RR(subjTest, t1)] = cpm_lr_D1D2_(mat_all(:,subjTrain, 4), mat_all(:,subjTest, 4), behav_all(subjTrain, t1), behav_all(subjTest, t1), thr);
                        
            [~, CPMpred_TM(subjTest, t1)] = cpm_lr_D1D2_(mat_all(:,subjTrain, t1), mat_all(:,subjTest, 5), behav_all(subjTrain, t1), behav_all(subjTest, t1), thr);
            [~, CPMpred_MM(subjTest, t1)] = cpm_lr_D1D2_(mat_all(:,subjTrain, 5), mat_all(:,subjTest, 5), behav_all(subjTrain, t1), behav_all(subjTest, t1), thr);
        end
    end
    
    % save CPMpred_* files
end

%% Assessing the model performance
for np = 1:nPerm
    % load CPMpred_* files
    
    % by correlation r
    for t1 = 1:3
        for t2 = 1:3
            [temp_R temp_P] = corr(behav_all(:,t2), squeeze(CPMpred_TT(:,t1,t2)));
            predCorr_TT.R(t1, t2, np) = temp_R';    predCorr_TT.P(t1,t2, np) = temp_P';    clear temp*
            [temp_R temp_P] = corr(behav_all(:,t2), squeeze(CPMpred_RT(:,t1,t2)));
            predCorr_RT.R(t1, t2, np) = temp_R';    predCorr_RT.P(t1,t2, np) = temp_P';    clear temp*
            [temp_R temp_P] = corr(behav_all(:,t2), squeeze(CPMpred_MT(:,t1,t2)));
            predCorr_MT.R(t1, t2, np) = temp_R';    predCorr_MT.P(t1,t2, np) = temp_P';    clear temp*
        
            [temp_R temp_P] = corr(behav_all(:,t2), squeeze(CPMpred_TR(:,t1)));
            predCorr_TR.R(t1, t2, np) = temp_R';    predCorr_TR.P(t1,t2, np) = temp_P';    clear temp*
            [temp_R temp_P] = corr(behav_all(:,t2), squeeze(CPMpred_RR(:,t1)));
            predCorr_RR.R(t1, t2, np) = temp_R';    predCorr_RR.P(t1,t2, np) = temp_P';    clear temp*

            [temp_R temp_P] = corr(behav_all(:,t2), squeeze(CPMpred_TM(:,t1)));
            predCorr_TM.R(t1, t2, np) = temp_R';    predCorr_TM.P(t1,t2, np) = temp_P';    clear temp*
            [temp_R temp_P] = corr(behav_all(:,t2), squeeze(CPMpred_MM(:,t1)));
            predCorr_MM.R(t1, t2, np) = temp_R';    predCorr_MM.P(t1,t2, np) = temp_P';    clear temp*
        end
    end
end
% save predCorr_* files


