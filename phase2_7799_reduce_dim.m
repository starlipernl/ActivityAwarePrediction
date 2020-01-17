% load prepared data from phase2_7799_load_data
load('prepared_data');

% Run dimensionality reduction algorithm per CV fold
cvtrain_feats_reduced = cell(1, numFolds);
cvvalid_feats_reduced = cell(1, numFolds);
cvtrain_cluster_inds = cell(1, numFolds);
cvvalid_cluster_inds = cell(1, numFolds);
parfor fold = 1:numFolds
    % assign training and validation data ccording to kfold indices
    CVtrainFeats = vertcat(trainFeats{idxCV ~= fold});
    CVtrainTarget = vertcat(trainTarget{idxCV ~= fold});
    CVvalidFeats = vertcat(trainFeats{idxCV == fold});
    CVvalidTarget = vertcat(trainTarget{idxCV == fold});
    
    % normalize training data 
    [cvtrain_features_Z, CVfeatures_mu, CVfeatures_sigma] = zscore(CVtrainFeats);
    [cvtrain_targetHR_Z, CVtargetHR_mu, CVtargetHR_sigma] = zscore(CVtrainTarget);
    % normalize test data with training transform parameters
    cvvalid_targetHR_Z = bsxfun(@rdivide, bsxfun(@minus, CVvalidTarget, CVtargetHR_mu), CVtargetHR_sigma);
    cvvalid_features_Z = bsxfun(@rdivide, bsxfun(@minus, CVvalidFeats, CVfeatures_mu), CVfeatures_sigma);
        
    % call dimensionality reduction and clustering functions
    [cvtrain_feats_reduced{fold}, cvvalid_feats_reduced{fold}] = dimReduce(cvtrain_features_Z(:, ...
        clusterFeatureInds), cvvalid_features_Z(:, clusterFeatureInds));
    [cvtrain_cluster_inds{fold}, cvvalid_cluster_inds{fold}] = runCluster(cvtrain_feats_reduced{fold}, ...
        cvvalid_feats_reduced{fold}, clusterNum);
end

% combine session data into training and testing matrices
train_features = vertcat(trainFeats{:});
train_target = vertcat(trainTarget{:});
test_features = vertcat(testFeats{:});
test_target = vertcat(testTarget{:});
% normalize training data 
[train_features_Z, features_mu, features_sigma] = zscore(train_features);
[train_targetHR_Z, targetHR_mu, targetHR_sigma] = zscore(train_target);
% normalize test data with training transform parameters
test_targetHR_Z = bsxfun(@rdivide, bsxfun(@minus, test_target, targetHR_mu), targetHR_sigma);
test_features_Z = bsxfun(@rdivide, bsxfun(@minus, test_features, features_mu), features_sigma);

%call dimensionality reduction and clustering functions
tic
[train_feats_red, test_feats_red] = dimReduce(train_features_Z(:, ...
    clusterFeatureInds), test_features_Z(:, clusterFeatureInds));
[train_cluster_inds_noshift, cluster_inds_noshift] = runCluster(train_feats_red, ...
    test_feats_red, clusterNum);
toc
save('features_reduced_all');

