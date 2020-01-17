% Load reduced features set and clustering indices
clc; clear;
load('features_reduced_all_new');
clusterNum = 2; %number of clusters (activities)
% last 2 columns are dHR d2HR defined  further below
featureGroupIDs = [1 2 3*ones(1,21) 4*ones(1,21), 5*ones(1,3)]; 
groupNames = {'Env. Temp', 'Humidity', 'Wrist accel', 'Chest accel', 'HR'};
currentSensor = [0.312 0.263, 0.263, 0.312];
current_weights = currentSensor/sum(currentSensor) * 10;
currentWrist = 4.126;
currentChest = 5.819;
currentTotal = currentWrist + currentChest;
numGroups = length(groupNames);
lambdas = exp(linspace(log(0.1), log(500), 500)); %group lasso parameter
% increaseAmounts = linspace(0,20,200); % RMSE increase for accuracy trade-off analysis
increaseAmountsP = [0 0.01 0.03 0.05 0.1 0.15 0.20]; %RMSE increase percentages for accuracy trade off analysis

%% run clustering if new number of clusters proposed

for fold = 1:numFolds     
    % call clustering functions
    [cvtrain_cluster_inds{fold}, cvvalid_cluster_inds{fold}] = runCluster(cvtrain_feats_reduced{fold}, ...
        cvvalid_feats_reduced{fold}, clusterNum);
end
[train_cluster_inds_noshift, cluster_inds_noshift] = runCluster(train_feats_red, ...
    test_feats_red, clusterNum);

 %% Preprocess heart rate signals to remove noise and spikes (optional)
trainTarget_raw = trainTarget;
testTarget_raw = testTarget;
trainFeatsHR_raw = trainFeatsHR;
testFeatsHR_raw = testFeatsHR;
for el = 1:length(trainTarget)
    target_temp = medfilt1(trainTarget{el}, 5);
    target_temp = movmean(target_temp, [4 0]);
    feature_temp = medfilt1(trainFeatsHR{el}(:,1),5);
    feature_temp = movmean(feature_temp, [4 0]);
    trainTarget{el} = target_temp;
    trainFeatsHR{el}(:,1) = feature_temp;
end
for el = 1:length(testTarget)
    target_temp = medfilt1(testTarget{el},5);
    target_temp = movmean(target_temp, [4 0]);
    feature_temp = medfilt1(testFeatsHR{el}(:,1),5);
    feature_temp = movmean(feature_temp, [4 0]);
    testTarget{el} = target_temp;
    testFeatsHR{el}(:,1) = feature_temp;
end

%% Cross Validation
tic
delh = 60;
cvtrain_c_inds_shift = cell(1, numFolds);
cvvalid_c_inds_shift = cell(1, numFolds);
for fold = 1:numFolds
    % assign training and validation data ccording to kfold indices
    CVtrainFeats = vertcat(trainFeats{idxCV ~= fold});
    CVtrainFeatsHR = vertcat(trainFeatsHR{idxCV ~= fold});
    CVtrainTarget = vertcat(trainTarget{idxCV ~= fold});
    CVtrainTime = vertcat(trainTime{idxCV ~= fold});
    CVvalidFeats = vertcat(trainFeats{idxCV == fold});
    CVvalidFeatsHR = vertcat(trainFeatsHR{idxCV == fold});
    CVvalidTarget = vertcat(trainTarget{idxCV == fold});
    CVvalidTime = vertcat(trainTime{idxCV == fold});
    % shift data by delh
    [CVtrainFeats_shift, CVtrainTarget_shift, cvtrain_c_inds_shift{fold}] ...
        = shiftVars(CVtrainTime, CVtrainFeats, CVtrainFeatsHR, ...
        CVtrainTarget, cvtrain_cluster_inds{fold},  delh);
    [CVvalidFeats_shift, CVvalidTarget_shift, cvvalid_c_inds_shift{fold}] = ...
        shiftVars(CVvalidTime, CVvalidFeats, CVvalidFeatsHR, ...
        CVvalidTarget, cvvalid_cluster_inds{fold}, delh);
    for c = 1:clusterNum
        % normalize training data 
        [cvtrain_features_Z{c}, CVfeatures_mu{c}, CVfeatures_sigma{c}] = ...
            zscore(CVtrainFeats_shift(cvtrain_c_inds_shift{fold} == c, :));
        [cvtrain_targetHR_Z{c}, CVtargetHR_mu{c}, CVtargetHR_sigma{c}] = ...
            zscore(CVtrainTarget_shift(cvtrain_c_inds_shift{fold} == c, :));
        % normalize test data with training transform parameters
        cvvalid_targetHR_Z{c} = bsxfun(@rdivide, bsxfun(@minus, ...
            CVvalidTarget_shift(cvvalid_c_inds_shift{fold} == c, :), ...
            CVtargetHR_mu{c}), CVtargetHR_sigma{c});
        cvvalid_features_Z{c} = bsxfun(@rdivide, bsxfun(@minus, ...
            CVvalidFeats_shift(cvvalid_c_inds_shift{fold} == c, :), ...
            CVfeatures_mu{c}), CVfeatures_sigma{c});
    end 
            
    for cluster = 1:clusterNum
        %Extract features for train and valid days for specific cluster
        clust_train_features_Z = cvtrain_features_Z{cluster};
        clust_train_targetHR_Z = cvtrain_targetHR_Z{cluster};
        clust_valid_features_Z = cvvalid_features_Z{cluster};
        clust_valid_targetHR = CVvalidTarget_shift(cvvalid_c_inds_shift{fold} == cluster, :);

        % orth CV train and CV test features (for group lasso) using SVD
        CV_train_features_Z_orth = zeros(size(clust_train_features_Z));
        CV_test_features_Z_orth = zeros(size(clust_valid_features_Z));
        %Singular value decomposition on on features matrix 
        for g = 1:max(featureGroupIDs)
            %disp(g);
            [CV_U, CV_S, CV_V] = svd(clust_train_features_Z(:, featureGroupIDs == g), 'econ'); % A = U * S * V' --- U = A * V * inv(S)
            if isequal(CV_S, 0)
                disp('bad S');
                CV_S = Inf;
            end
            CV_train_features_Z_orth(:, featureGroupIDs == g) = CV_U;
            %project test features onto orthonormal basis of train
            %features
            CV_test_features_Z_orth(:, featureGroupIDs == g) = clust_valid_features_Z(:, featureGroupIDs == g) * CV_V / CV_S;
        end

        %Group lasso
        CV_b_grp_HR = zeros(length(featureGroupIDs), length(lambdas));
        % initialize parameters
        CV_actualHR = clust_valid_targetHR; %+ CV_test_features(:, 52);
        % cycle through range of lamda (regularization hyperparamter)
        for lambdaInd = 1:length(lambdas)
            % Train group lasso (returns regression coefficients)
            CV_b_grp_HR(:, lambdaInd) = grplassoShooting( ... 
                CV_train_features_Z_orth, clust_train_targetHR_Z, ...
                featureGroupIDs, lambdas(lambdaInd), 2e4, 1e-10, false);
            % Test group lasso on HR
            CV_predicted_HR = (CV_test_features_Z_orth * CV_b_grp_HR(:, lambdaInd))...
                * CVtargetHR_sigma{cluster} + CVtargetHR_mu{cluster}; %+ CV_test_features(:,52);
            %smooth_CV_predicted_BR = movmean(CV_predicted_BR, [4 0]);
            err = CV_predicted_HR - CV_actualHR;
            %err = smooth_CV_predicted_BR - CV_actualBR;
            %Mean square error per cv test day per lambda value
            CV_MSE_HR{fold}{cluster}(lambdaInd) = rms(err) ^ 2; %CV_MSE_HR + CV_nump_cluster/CV_nump * rms(err) ^ 2;
            CV_RMSE_HR{fold}{cluster}(lambdaInd) = rms(err);
            CV_NRMSE_HR{fold}{cluster}(lambdaInd) = rms(err)/range(CV_actualHR);
%             CV_MSE_HR{cluster}{fold}(lambdaInd) = rms(err) ^ 2;
%             CV_RMSE_HR{cluster}{fold}(lambdaInd) = rms(err);
%             CV_NRMSE_HR{cluster}{fold}(lambdaInd) = rms(err)/range(CV_actualHR);
        end
    end 
end
for cluster = 1:clusterNum
    %Calculate average MSE across CV folds for each lambda and
    %find lambda that minimizes MSE
    [avg_CV_MSE_HR{cluster}, avg_CV_RMSE_HR{cluster}, ... 
        avg_CV_NRMSE_HR{cluster}] = deal(zeros(1,size(lambdas,2)));
    for fold = 1:numFolds
        avg_CV_MSE_HR{cluster} = avg_CV_MSE_HR{cluster} +  ...
            CV_MSE_HR{fold}{cluster} * ... 
            (sum(cvvalid_c_inds_shift{fold} == cluster) / ...
            length(cvvalid_c_inds_shift{fold}));
        avg_CV_RMSE_HR{cluster} = avg_CV_RMSE_HR{cluster} +  ...
            CV_RMSE_HR{fold}{cluster} * ... 
            (sum(cvvalid_c_inds_shift{fold} == cluster) / ...
            length(cvvalid_c_inds_shift{fold}));
        avg_CV_NRMSE_HR{cluster} = avg_CV_NRMSE_HR{cluster} +  ...
            CV_NRMSE_HR{fold}{cluster} * ... 
            (sum(cvvalid_c_inds_shift{fold} == cluster) / ...
            length(cvvalid_c_inds_shift{fold}));
    end
    [avg_CV_RMSE_min{cluster}, optimal_lambda{cluster}] = min(avg_CV_RMSE_HR{cluster});
%     avg_CV_MSE_HR{cluster} = mean(vertcat(CV_MSE_HR{cluster}{:}), 1);
%     avg_CV_RMSE_HR{cluster} = mean(vertcat(CV_RMSE_HR{cluster}{:}), 1); 
%     avg_CV_NRMSE_HR{cluster} = mean(vertcat(CV_NRMSE_HR{cluster}{:}), 1);
%     [avg_CV_RMSE_min{cluster}, optimal_lambda{cluster}] = min(avg_CV_RMSE_HR{cluster});
end 
toc

%% Train and test model
% combine session data 
train_features = vertcat(trainFeats{:});
train_target = vertcat(trainTarget{:});
train_featuresHR = vertcat(trainFeatsHR{:});
train_time = vertcat(trainTime{:});
test_features = vertcat(testFeats{:});
test_target = vertcat(testTarget{:});
test_featuresHR = vertcat(testFeatsHR{:});
test_time = vertcat(testTime{:});

% shift data by delh
[train_feats_shift, train_target_shift, train_cluster_inds] ...
    = shiftVars(train_time, train_features, train_featuresHR, ...
    train_target, train_cluster_inds_noshift,  delh);
[test_feats_shift, test_target_shift, cluster_inds] ...
    = shiftVars(test_time, test_features, test_featuresHR, ...
    test_target, cluster_inds_noshift,  delh);
[train_features_Z, train_targetHR_Z, test_targetHR_Z, test_features_Z] ...
    = deal(cell(1,clusterNum));
[features_mu, features_sigma, targetHR_mu, targetHR_sigma] ...
    = deal(cell(1,clusterNum));
for c = 1:clusterNum
    % normalize training data 
    [train_features_Z{c}, features_mu{c}, features_sigma{c}] = ...
        zscore(train_feats_shift(train_cluster_inds == c, :));
    [train_targetHR_Z{c}, targetHR_mu{c}, targetHR_sigma{c}] = ...
        zscore(train_target_shift(train_cluster_inds == c, :));
    % normalize test data with training transform parameters
    test_targetHR_Z{c} = bsxfun(@rdivide, bsxfun(@minus, ...
        test_target_shift(cluster_inds == c, :), targetHR_mu{c}), targetHR_sigma{c});
    test_features_Z{c} = bsxfun(@rdivide, bsxfun(@minus, ...
        test_feats_shift(cluster_inds == c, :), features_mu{c}), features_sigma{c});
end

%% Test models 
tic
for cluster = 1:clusterNum
    train_features_Zc = train_features_Z{cluster};
    train_targetHR_Zc = train_targetHR_Z{cluster};
    test_targetc = test_target_shift(cluster_inds == cluster, :);
    test_features_Zc{cluster} = test_features_Z{cluster};

    % orthonormal train and test features (for group lasso)
    train_features_Z_orth{cluster} = [];
    test_features_Z_orth = [];
    for g = 1:max(featureGroupIDs)
        %disp(g);
        [CV_U, CV_S, CV_V] = svd(train_features_Zc(:, featureGroupIDs == g), 'econ'); % A = U * S * V' --- U = A * V * inv(S)
        if isequal(CV_S, 0)
            disp('bad S 2');
            CV_S = Inf; 
        end
        train_features_Z_orth{cluster}(:, featureGroupIDs == g) = CV_U;
        test_features_Z_orth(:, featureGroupIDs == g) = test_features_Zc{cluster}(:, featureGroupIDs == g) * CV_V / CV_S;
    end
    %b_grp_BR = [];
    for lambdaInd = 1:length(lambdas)
        % Train group lasso
        b_grp_HR = grplassoShooting(train_features_Z_orth{cluster}, train_targetHR_Zc, featureGroupIDs, lambdas(lambdaInd), 2e4, 1e-10, false);

        % Test group lasso on BR
        predicted_HR{cluster}(:,lambdaInd) = (test_features_Z_orth * b_grp_HR) ...
            * targetHR_sigma{cluster} + targetHR_mu{cluster};
        %smooth_CV_predicted_BR = movmean(predicted_BR, [4 0]);
        actualHR = test_targetc;
        err = predicted_HR{cluster}(:,lambdaInd) - actualHR;
        %err = smooth_CV_predicted_BR - CV_actualBR;
        MSE_HR{cluster}(lambdaInd) = rms(err) ^ 2;
        RMSE_HR{cluster}(lambdaInd) = rms(err);
        NRMSE_HR{cluster}(lambdaInd) = sqrt(MSE_HR{cluster}(lambdaInd)) / range(actualHR);
        numFeatures_HR{cluster}(lambdaInd) = sum(b_grp_HR ~= 0);
        %sensorGroupsUsed{cluster}(lambdaInd) =
        %unique(featureGroupIDs(b_grp_BR ~= 0)); 
        sensorGroupsUsed{cluster}(lambdaInd, 1:max(featureGroupIDs)) = false;
        sensorGroupsUsed{cluster}(lambdaInd, unique(featureGroupIDs(b_grp_HR ~= 0))) = true;
        numSensors_HR{cluster}(lambdaInd) = length(unique(featureGroupIDs(b_grp_HR ~= 0)));
    end
end


% average across clusters (with each cluster's optimal lambda)
avg_MSE_HR = 0;
avg_RMSE_HR = 0;
for cluster = 1:clusterNum
    avg_MSE_HR = avg_MSE_HR + MSE_HR{cluster}(optimal_lambda{cluster}) * sum(cluster_inds == cluster);
    avg_RMSE_HR = avg_RMSE_HR + RMSE_HR{cluster}(optimal_lambda{cluster}) * sum(cluster_inds == cluster);
    avg_RMSE_HR_cluster(cluster) = RMSE_HR{cluster}(optimal_lambda{cluster});  
end
avg_MSE_HR = avg_MSE_HR / length(cluster_inds);
avg_RMSE_HR = avg_RMSE_HR / length(cluster_inds);
avg_NRMSE_HR = avg_RMSE_HR/range(test_target_shift);
    
disp('Finished');
toc

%% Find some average error values

actual_testHR = test_target_shift;
minLambda_predicted_HR = zeros(size(actual_testHR));
err_final_rms_cluster = zeros(1, clusterNum);
[c_num, c_p] = deal(cell(clusterNum, 1));
for cluster = 1:clusterNum
    avg_RMSE_HR_cluster(cluster,:) = RMSE_HR{cluster}(optimal_lambda{cluster}); 
    minLambda_predicted_HR(cluster_inds == cluster) = ...
        predicted_HR{cluster}(:,optimal_lambda{cluster});
    err_final_rms_cluster(cluster) = rms(minLambda_predicted_HR(cluster_inds == cluster) ...
        - actual_testHR(cluster_inds == cluster));
    c_num{cluster,:} = sum(cluster_inds == cluster);
    c_p{cluster,:} = c_num{cluster}/length(cluster_inds);
end

err_final = minLambda_predicted_HR - actual_testHR;
results.err = err_final;
results.rms = rms(err_final);
results.nrms = results.rms/range(actual_testHR);
results.rmsr = rms(err_final./actual_testHR);
results.mae = sum(abs(err_final))/length(err_final);
results.mape = sum(abs(err_final./actual_testHR))/length(err_final);
results.r2 = 1 - sum(err_final.^2)/sum((actual_testHR-mean(actual_testHR)).^2);
% res_cp.err = actual_testHR - mean(actual_testHR);
res_cp.err = actual_testHR - test_feats_shift(:,45);
res_cp.rms = rms(res_cp.err);
res_cp.nrms = res_cp.rms/range(actual_testHR);
res_cp.rmsr = rms(res_cp.err./actual_testHR);
res_cp.mae = sum(abs(res_cp.err))/length(actual_testHR);
res_cp.mape = sum(abs(res_cp.err./actual_testHR))/length(actual_testHR);
res_cp.r2 = 1 - sum(res_cp.err.^2)/sum((actual_testHR-mean(actual_testHR)).^2);



%% Get list of features used per test day and cluster and optimal features
%used for lambda with min RMSE

clear optimalFeaturesUsed featuresUsed optimalsensorGroupsUsed;
groupInds = 1:max(featureGroupIDs);
currentReduction_avg = 0;
[featuresUsed, optimalFeaturesUsed, optimalsensorGroupsUsed] = ...
    deal(cell(1, clusterNum));
currentReduction = zeros(2, clusterNum);
for cluster = 1:clusterNum
    %initialize
    featuresUsed{cluster} = false(length(lambdas), length(featureGroupIDs));
    for lambdaInd = 1:length(lambdas)
        for groupInd = groupInds(sensorGroupsUsed{cluster}(lambdaInd, 1:4))
            enabledFeatures = featureGroupIDs == groupInd;
            featuresUsed{1, cluster}(lambdaInd, enabledFeatures) = true;
        end
    end
    %features used with optimal lambda
    lambdaInd = optimal_lambda{cluster};
    optimalFeaturesUsed{cluster} = featuresUsed{cluster}(lambdaInd, :);
    optimalsensorGroupsUsed{cluster} = sensorGroupsUsed{cluster}(lambdaInd, 1:4);
    currentReduction(1, cluster) = sum(currentSensor .* not(optimalsensorGroupsUsed{cluster}));
    currentReduction(2, cluster) = currentReduction(1, cluster)/currentTotal;
    currentReduction_avg(1, :) = currentReduction_avg + currentReduction(1, cluster)*c_p{cluster};
end
currentReduction_avg(2, :) = currentReduction_avg(1, :)/currentTotal;

return;

%% SVM run training and testing 
% for SVM HR prediction testing for comparison to Group lasso

% Set the svm run flag:
% 1 - Predict Change in HR using lasso optimal features
% 2 - Predict Change in HR using all features
% 3 - Predict Actual HR using lasso optimal features
% 4 - Predict Actual HR using all features
% svm_run_flag = 1;
% switch svm_run_flag
%     case 1 
%         svm;
%     case 2
%         svm_allFeats
%     case 3
%         svm_hr
%     case 4
%         svm_allFeats_hr
% end

for i = 1:2
    if i == 1
       svm;
    else
       svm_allFeats;
    end
    svm_actual_HR_full = vertcat(svm_actual_HR{1:clusterNum});
    svm_predicted_HR_full = vertcat(svm_predicted_HR{1:clusterNum});
    svm_err = svm_actual_HR_full-svm_predicted_HR_full;
    svm_rms_final = rms(svm_err);
    svm_nrms = svm_rms_final/range(svm_actual_HR_full);
    svm_rmsr = rms(svm_err./svm_actual_HR_full);
    svm_mae = sum(abs(svm_err))/length(svm_err);
    svm_mape = sum(abs(svm_err./svm_actual_HR_full))/length(svm_err);
    svm_r2 = 1 - sum(svm_err.^2)/sum((svm_actual_HR_full-mean(svm_actual_HR_full)).^2);
    if i == 1
        svm_results = struct('rmse', svm_rms_final, 'nrmse', svm_nrms, 'mae', svm_mae, 'mape', svm_mape, 'r2', svm_r2);
    else
        svm_results_allFeats = struct('rmse', svm_rms_final, 'nrmse', svm_nrms, 'mae', svm_mae, 'mape', svm_mape, 'r2', svm_r2);
    end
end
toc


%% Plot feature/sensor disappearance
figure
clf
% groupNames = {'Env. Temp', 'Humidity', 'Wrist accel', 'Chest accel', 'Heart Rate'};

for cluster = 1:clusterNum
%     h = subplot(2, 3, cluster);
    h = subplot(1, clusterNum, cluster);
    feature_exist = featuresUsed{cluster}';
    feature_exist = [ones(size(feature_exist, 1), 1) feature_exist];
    feature_exist(:, end) = 0;
    feature_lost_table = diff(feature_exist, 1, 2);
    feature_lost_table = feature_lost_table([1 2 3 24], :);
    [feature_lost, lambda_lost] = find(feature_lost_table == -1);
    feature_lost = flipud(feature_lost);
    lambda_lost = flipud(lambda_lost);
    [feature_lost_latest, idxa, idxb] = unique(feature_lost, 'stable');
    lambda_lost_latest = lambda_lost(idxa);
    feature_lost_latest = flipud(feature_lost_latest);
    lambda_lost_latest = flipud(lambda_lost_latest);
    [feature_lost_latest, sort_idx] = sort(feature_lost_latest);
    lambda_lost_latest = lambda_lost_latest(sort_idx);
    sensor_lost = groupNames(feature_lost_latest);
    
    hold on;
    minLambdaInd = optimal_lambda{cluster};
    bar(find(lambda_lost_latest < minLambdaInd), lambdas(lambda_lost_latest(lambda_lost_latest < minLambdaInd)), 'FaceColor', [1 0.5 0.3]);
    bar(find(lambda_lost_latest >= minLambdaInd), lambdas(lambda_lost_latest(lambda_lost_latest >= minLambdaInd)), 'FaceColor', [0.3 1 0.5]);
%     bar(1:sum(lambda_lost_latest < minLambdaInd), lambdas(lambda_lost_latest(lambda_lost_latest < minLambdaInd)), 'FaceColor', [1 0.5 0.3]);
%     bar((numGroups+1)-sum(lambda_lost_latest >= minLambdaInd):numGroups, lambdas(lambda_lost_latest(lambda_lost_latest >= minLambdaInd)), 'FaceColor', [0.3 1 0.5]);
    plot([0 numGroups+1], lambdas(minLambdaInd) * ones(2, 1), 'k--');

    set(h, 'yscale', 'log')
    set(h, 'FontSize', 12);
    set(h, 'XLim', [0 numGroups+1]);
    set(h, 'YLim', [lambdas(1) / 1.1 lambdas(end)]);
    grid on;
    ylabel('\lambda');
    title(['Cluster = ' num2str(cluster)]);
   % title(['No Clustering']);
    xticks(1:length(feature_lost_latest));
%     xtickangle(45);
%     xticklabels(sensor_lost);
     xticklabels({'A', 'B', 'C', 'D', 'E'});

end


%% Plot RMSE and numSensors
figure;
clf;

for cluster = 1:clusterNum
%         lambdaInd = optimal_lambda{cluster};
    %subplot(length(test_section_days), clusterNum, (test_section_day-1)*clusterNum + cluster);
    subplot(1, clusterNum, cluster);
    yyaxis left
    ylim([-inf max(max(sqrt(avg_CV_MSE_HR{cluster})),max(sqrt(MSE_HR{cluster})))]);
    hold on;
    plot(lambdas, sqrt(avg_CV_MSE_HR{cluster}), 'r');
    plot(lambdas, sqrt(MSE_HR{cluster}), 'b-');
    xlabel('\lambda')
    ylabel('RMSE');
    xticks([0.01 1 10 30])
    yyaxis right
    ylim([0 numGroups]) 
    yticks([0 numGroups])
    plot(lambdas, numSensors_HR{cluster}, 'k--');
    ylabel('Number of Sensors');
    rmsPlotTitle = ['Cluster = ' num2str(cluster)];
    title(rmsPlotTitle);
    ax = gca;
    ax.YAxis(2).Color = 'black';

    set(gca, 'xscale', 'log')
    set (gca, 'fontsize', 10)
    set(gca, 'XLim', [lambdas(1) / 1.1 lambdas(end)]);
end

rmsLgd = legend('CV RMSE', 'Test RMSE', 'Num Sensors');
rmsLgd.FontSize = 8;


%% Plot NRMSE and numSensors
figure(2);
clf;

for cluster = 1:clusterNum
    subplot(1, clusterNum, cluster);
    yyaxis left
    ylim([-inf max(max(avg_CV_NRMSE_HR{cluster},max(NRMSE_HR{cluster})))]);
    hold on;
    plot(lambdas, (avg_CV_NRMSE_HR{cluster}), 'b');
    plot(lambdas, (NRMSE_HR{cluster}), 'b--');
    xlabel('\lambda')
    ylabel('RMSE');
    xticks([0.01 1 10 40])
    yyaxis right
    ylim([0 numGroups]) 
    yticks([0 numGroups])
    plot(lambdas, numSensors_HR{cluster});
    ylabel('Num Sensors');

    set(gca, 'xscale', 'log')
    set (gca, 'fontsize', 12)
    set(gca, 'XLim', [lambdas(1) / 1.1 lambdas(end)]);

end
legend('CV NRMSE', 'Test NRMSE', 'Num Sensors');

% %% Plot some avg CV curves
% 
% 
% test_section_day = 1
% cluster = 1
% 
% figure
% clf
% 
% yyaxis left
% hold on;
% plot(lambdas, avg_CV_MSE_HR{cluster}, 'b');
% plot(lambdas, MSE_HR{test_section_day}{cluster}, 'r');
% 
% yyaxis right
% plot(lambdas, numSensors_HR{test_section_day}{cluster});
% 
% set(gca, 'xscale', 'log')
% 
% %% Plot some avg CV NRMSE curves
% 
% 
% test_section_day = 1
% cluster = 1
% 
% figure
% clf
% 
% yyaxis left
% hold on;
% plot(lambdas, avg_CV_NRMSE_HR{test_section_day}{cluster}, 'b');
% plot(lambdas, NRMSE_HR{test_section_day}{cluster}, 'r');
% 
% yyaxis right
% plot(lambdas, numSensors_HR{test_section_day}{cluster});
% 
% set(gca, 'xscale', 'log')
% 
% 
% 
%% Visualize train data clusters 
figure
clf;
hold on;
legtxt = [];


for cluster = 1:clusterNum
    plotData = train_feats_red(train_cluster_inds_noshift == cluster, :);
    h = scatter3(plotData(:,1), plotData(:,2), plotData(:,3), '.');
end
legend(legtxt);
legend(['1 (' num2str(sum(train_cluster_inds == 1)) ')'],...
['2 (' num2str(sum(train_cluster_inds == 2)) ')'],...
['3 (' num2str(sum(train_cluster_inds == 3)) ')'],...
['4 (' num2str(sum(train_cluster_inds == 4)) ')'],...
['5 (' num2str(sum(train_cluster_inds == 3)) ')'],...
['6 (' num2str(sum(train_cluster_inds == 3)) ')']);
title('Clustering Visualization');
axis equal;
set(gca, 'FontSize', 12);
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
set(gca,'ZTickLabel',[]);
% 
%% Visualize testing cluster over time 
% if clusterNum == 1
%     cluster_file = ['results\K_sqrtP\delHR_' num2str(delh) 's_nocluster.mat'];
% else
%     cluster_file = ['results\K_sqrtP\delHR_' num2str(delh) 's.mat'];
% end
% load(cluster_file, 'features_Z_reduced');
% clear new_clusters
% new_clusters = zeros(length(cluster_inds),1);
% new_clusters(cluster_inds == 1, :) = 2;
% new_clusters(cluster_inds == 2, :) = 3;
% new_clusters(cluster_inds == 3, :) = 1;
% new_clusters(cluster_inds == 4, :) = 4;
% modefun = @(x) mode(x(:));
% cluster_mode = nlfilter(cluster_inds, [20 1], modefun);
smoothHRpredict = minLambda_predicted_HR;%movmean(minLambda_predicted_HR, [0 4]);
smoothactual_testHR = medfilt1(actual_testHR, 5);
figure;
t_plot = 1:length(cluster_inds);
subplot(3,1,1); hold on
for ii = 1:3
h1= plot(t_plot/60, test_feats_red(:, ii)); 
%legend('IMU Feature')
end
hold off
ylabel('Accel');
set(gca, 'fontsize', 10)
hold off
h2 = subplot(3,1,2); plot(t_plot/60, cluster_inds, '.', 'MarkerSize', 10); 
ylabel('Cluster Number');
set(gca, 'fontsize', 10)
%legend('Cluster');
subplot(3,1,3); hold on
h3 = plot(t_plot, actual_testHR, t_plot, smoothHRpredict, '--'); 
xtickformat('%d');
legend('Measured', 'Predicted');
hold off
ylabel('HR (bpm)')
xlabel('t (mins)')
set(gca, 'fontsize', 10)
%lgd = legend([h1, h2, h3, h4],{'IMU Feature', 'Cluster', 'Measure HR', 'Predicted HR'});
%set(gca, 'YTicks', [0 1 2 3 4 5])

%% IMU Windows per cluster

cluster_file = ['results\K_sqrtP\delHR_' num2str(delh) 's_nocluster.mat'];
load(cluster_file, 'features_Z_reduced');
figure;
time = 1:length(cluster_inds);
cluster_inds_plot = {[31:60], [2315:2344], [1321:1350], [526:555]};
for cluster=1:4
subplot(2,2,cluster); hold on
for ii = 1:3
plot(features_Z_reduced(cluster_inds_plot{cluster}, ii)); 
%legend('IMU Feature')
end
hold off
ylabel('Accel');
xlabel('time (s)');
title(['Cluster = ' num2str(cluster)]);
set(gca, 'fontsize', 10)
end

%% Find the lambda indices given by the increase allowed RMSE amount vector
increaseAmounts = linspace(0,20,100);
for test_section_day = test_section_days
    for cluster = 1:clusterNum
        increaseHRAmountLambdaInd{cluster}(1) = optimal_lambda{cluster};
        for increaseAmountInd = 2:length(increaseAmounts)
            increaseAmount = increaseAmounts(increaseAmountInd);
            %Find index of lambda with RMSE closest to increased RMSE 
            if increaseHRAmountLambdaInd{cluster}(increaseAmountInd-1) == 300
                increaseHRAmountLambdaInd{cluster}(increaseAmountInd) = 300;
            elseif any(sqrt(avg_CV_MSE_HR{cluster}(increaseHRAmountLambdaInd{cluster}(increaseAmountInd-1):end))<=sqrt(avg_CV_MSE_HR{cluster}(increaseHRAmountLambdaInd{cluster}(increaseAmountInd-1))))
%                 [~, increaseHRAmountLambdaInd{cluster}(increaseAmountInd)] = min(...
%                         (sqrt(avg_CV_MSE_HR{cluster}(increaseHRAmountLambdaInd{cluster}(increaseAmountInd-1):end)) - ...
%                         (sqrt(avg_CV_MSE_HR{cluster}(optimal_lambda{cluster}))+increaseAmount)));
                [~, increaseHRAmountLambdaInd{cluster}(increaseAmountInd)] = find(...
                        (sqrt(avg_CV_MSE_HR{cluster}(increaseHRAmountLambdaInd{cluster}(increaseAmountInd-1):end))<= ...
                        (sqrt(avg_CV_MSE_HR{cluster}(optimal_lambda{cluster}))+increaseAmount)), 1, 'last');
                increaseHRAmountLambdaInd{cluster}(increaseAmountInd) = ...
                        increaseHRAmountLambdaInd{cluster}(increaseAmountInd) + increaseHRAmountLambdaInd{cluster}(increaseAmountInd-1)-1;
            else
                [~, increaseHRAmountLambdaInd{cluster}(increaseAmountInd)] = min(...
                    abs(sqrt(avg_CV_MSE_HR{cluster}(increaseHRAmountLambdaInd{cluster}(increaseAmountInd-1):end)) - ...
                    (sqrt(avg_CV_MSE_HR{cluster}(optimal_lambda{cluster}))+increaseAmount)));
                increaseHRAmountLambdaInd{cluster}(increaseAmountInd) = ...
                    increaseHRAmountLambdaInd{cluster}(increaseAmountInd) + increaseHRAmountLambdaInd{cluster}(increaseAmountInd-1);
    %             [~, increaseHRAmountLambdaInd{cluster}(increaseAmountInd)] = min(...
    %                 abs(sqrt(avg_CV_MSE_HR{cluster}(optimal_lambda{cluster}+1:end)) - ...
    %                 (sqrt(avg_CV_MSE_HR{cluster}(optimal_lambda{cluster}))+increaseAmount)));
    %             increaseHRAmountLambdaInd{cluster}(increaseAmountInd) = ...
    %                 increaseHRAmountLambdaInd{cluster}(increaseAmountInd) + optimal_lambda{cluster};
        
            end
        end
%         increaseHRAmountLambdaInd{cluster}(1) = ...
%                increaseHRAmountLambdaInd{cluster}(1) - 1;
     end
end


%% Find the lambda indices given by the increase allowed RMSE PERCENT amount vector
% % PERCENT RMSE
% for test_section_day = test_section_days
%     for cluster = 1:clusterNum
%         for increaseAmountInd = 1:length(increaseAmountsP)
%             increaseAmount = increaseAmountsP(increaseAmountInd);
%             %Find index of lambda with RMSE closest to increased RMSE 
%             [~, increaseHRAmountLambdaInd{cluster}(increaseAmountInd)] = min(...
%                 abs(sqrt(avg_CV_MSE_HR{cluster}(optimal_lambda{cluster}+1:end)) - ...
%                 (1+increaseAmount)*sqrt(avg_CV_MSE_HR{cluster}(optimal_lambda{cluster})))...
%                 );
%             increaseHRAmountLambdaInd{cluster}(increaseAmountInd) = ...
%                 increaseHRAmountLambdaInd{cluster}(increaseAmountInd) + optimal_lambda{cluster};
%             %         [~, increaseHRAmountLambdaInd{cluster}(increaseAmountInd)] = min(abs(CV_avg_RMSE_HR{cluster}(minHRCVLambdaInd+1:end) - (1+increaseAmount)*min(CV_avg_RMSE_HR{cluster}(minHRCVLambdaInd+1:end))));
%             %         increaseHRAmountLambdaInd{cluster}(increaseAmountInd) = increaseHRAmountLambdaInd{cluster}(increaseAmountInd) + minHRCVLambdaInd;
%         end
%         increaseHRAmountLambdaInd{cluster}(1) = ...
%                 increaseHRAmountLambdaInd{cluster}(1) - 1;
%      end
% end


%% Sensors used for each day using new allowable increased RMSE value 
clear current_reduction_new;
for test_section_day = test_section_days
    for increaseInd = 1:length(increaseAmounts)
        sensors_used_results{increaseInd} = false(max(featureGroupIDs), clusterNum);
        for cluster = 1:clusterNum
    %         lambdaInd = optimal_lambda{cluster};
            %increaseAmountInd = find(increaseAmounts == RMSE_increase);
            increaseAmountInd = increaseInd;
            lambdaInd = increaseHRAmountLambdaInd{cluster}(increaseAmountInd);

            %sensors_used_results(sensorGroupsUsed{cluster}(lambdaInd), (test_section_day-1) * 4 + cluster) = true;
            sensors_used_results{increaseInd}(:, cluster) = sensorGroupsUsed{cluster}(lambdaInd, :)';
            %unique(featureGroupIDs(b_grp_HR ~= 0))
        end
        current_reduction_new(increaseInd, 1:clusterNum) = currentSensor * not(sensors_used_results{increaseInd});
        current_reduction_new(increaseInd, clusterNum + 1) = 0;
        for cluster = 1:clusterNum
            current_reduction_new(increaseInd, clusterNum + 1) = current_reduction_new(increaseInd, clusterNum + 1) + current_reduction_new(increaseInd, cluster)*c_p{cluster};
        end
    end
end

% sensors_used_results;
% 

%% Calculate average RMSE, number sensors, and cluster counts using the new
%  increased allowable RMSE percentage lambda index (lambdaInc)
RMSE_increase = 0.03;
disp('RMSE for days');
RMSE_results = [];
avg_MSE_HR_it = [];
for test_section_day = section_days
    %disp(['Day ' num2str(test_section_day)]);
    %MSE_HR{cluster}(lambdaInd)

    %avg_MSE_HR_it(test_section_day) = 0;
    avg_MSE_HR_it = 0;
    avg_numSensors_HR = 1;
    for cluster = 1:clusterNum
        %disp(['Cluster ' num2str(cluster)]);
%         lambdaInd = optimal_lambda{cluster};
        increaseAmountInd = find(increaseAmounts == RMSE_increase);
        lambdaInd = increaseHRAmountLambdaInd{cluster}(increaseAmountInd);
        RMSE_results(test_section_day * 3 - 2, cluster) = sqrt(MSE_HR{cluster}(lambdaInd));
        RMSE_results(test_section_day * 3 - 1, cluster) = numSensors_HR{cluster}(lambdaInd);
        RMSE_results(test_section_day * 3    , cluster) = sum(cluster_inds == cluster);
        
        %avg_MSE_HR_it(test_section_day) = avg_MSE_HR_it(test_section_day) + MSE_HR{cluster}(lambdaInd) * sum(cluster_inds == cluster);
        avg_MSE_HR_it = avg_MSE_HR_it + MSE_HR{cluster}(lambdaInd) * sum(cluster_inds == cluster);
        avg_numSensors_HR = avg_numSensors_HR + numSensors_HR{cluster}(lambdaInd) * sum(cluster_inds == cluster);
        %disp([num2str(sum(cluster_inds == cluster)) ' ' num2str(MSE_HR{cluster}(lambdaInd))]);
    end
    %avg_MSE_HR_it(test_section_day) = avg_MSE_HR_it(test_section_day) / length(cluster_inds);
    avg_MSE_HR_it = avg_MSE_HR_it / length(cluster_inds);
    avg_numSensors_HR = avg_numSensors_HR / length(cluster_inds);
    avg_clusterNum = length(cluster_inds) / clusterNum;

    RMSE_results(test_section_day * 3 - 2, clusterNum+1) = sqrt(avg_MSE_HR_it);
    RMSE_results(test_section_day * 3 - 1, clusterNum+1) = avg_numSensors_HR;
    RMSE_results(test_section_day * 3    , clusterNum+1) = avg_clusterNum;

end
avg_RMSE_HR_it = sqrt(avg_MSE_HR_it);
avg_RMSE_HR_it = round(avg_RMSE_HR_it' * 1000) / 1000;
RMSE_results = round(RMSE_results * 1000) / 1000;
%disp(avg_avg_MSE_HR);

%% Try different dimensionality reduction techniques 

% close all;
% clc;
% 
% techniqueNames = {'PCA', 'LDA', 'MDS', 'ProbPCA', 'FactorAnalysis', 'GPLVM', ...
%     'Sammon', 'Isomap', 'LandmarkIsomap', 'LLE', 'Laplacian', 'HessianLLE', ...
%     'LTSA', 'MVU', 'CCA', 'LandmarkMVU', 'FastMVU', 'DiffusionMaps', ...
%     'KernelPCA', 'GDA', 'SNE', 'SymSNE', 'tSNE', 'LPP', 'NPE', 'LLTSA', ...
%     'SPE', 'Autoencoder', 'LLC', 'ManifoldChart', 'CFA', 'NCA', 'MCML', 'LMNN'};
techniqueNames = {'tSNE'};
% clusterFeatureInds = 5:46; % 5:46 5:25
% skip 6:9 12:17 21:23 29:31 33:34
for techniqueNamesInd = [1]
%     disp([num2str(techniqueNamesInd) '/' num2str(numel(techniqueNames)) ' - ' techniqueNames{techniqueNamesInd}]);
%     [accelDataReduced3, mapping] = compute_mapping(features_Z(:, clusterFeatureInds), techniqueNames{techniqueNamesInd}, 3, 21, 30);
%     [accelDataReduced2, mapping] = compute_mapping(features_Z(:, clusterFeatureInds), techniqueNames{techniqueNamesInd}, 2, train_sections_features_Z_reduced, 30);
%     epsilon = 0.009;
%     MinPts = 50;
%     %DBSCAN clustering function found online
%     trainClusterInd = DBSCAN(accelDataReduced, epsilon, MinPts);
%     clusterCount = max(trainClusterInd);
    [train_sections_features_Z_tsne, mapping_tsne] = compute_mapping([train_features_Z(:, clusterFeatureInds); valid_features_Z(:, clusterFeatureInds)], 'tSNE', 3);
    accelDataReduced3 = train_section_features_Z_tsne(size(train_features_Z(:, clusterFeatureInds), 1)+1:end,:);
    train_section_features_Z_tsne = train_section_features_Z_tsne(1:size(train_features_Z(:, clusterFeatureInds)), :);
    [train_cluster_inds_tsne, cluster_centroids_tsne] = kmeans(train_sections_features_Z_tsne, clusterNum);
    trainclusterSize = [];
    for cluster = 1:clusterNum
        trainclusterSize(cluster) = sum(trainClusterInd_tsne == cluster);
    end
    [~, ind] = sort(trainclusterSize, 'descend');
    new_cluster_inds = [];
    for cluster = 1:clusterNum
        new_cluster_inds(trainClusterInd_tsne == ind(cluster)) = cluster;
    end
    trainClusterInd_tsne = new_cluster_inds;
    accelDataReduced3;
    clusterInd_tsne = kmeans(accelDataReduced3, 4, 'Start', cluster_centroids_tsne);
    clusterCount = 4;
    figure
    hold on;
    for cluster = 1:clusterCount
%         plotData = accelDataReduced3(trainClusterInd == cluster, :);
        plotData = accelDataReduced3(clusterInd_tsne == cluster, :);
        plot3(plotData(:,1), plotData(:,2), plotData(:,3), '.', 'MarkerSize', 6);
%          plot(plotData(:,1), plotData(:,2), '.', 'MarkerSize', 6);
    end
    title(techniqueNames{techniqueNamesInd});
    legend(['none' cellfun(@num2str, num2cell(1:clusterCount), 'uniform', 0)]);
%     saveas(gcf,['reduced_' num2str(techniqueNamesInd) '.png']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% kmeans elbow method
features_red_AE = load('phase2_featsred_02122019', 'features_red');
clust = zeros(size(features_reduced2d{5},1),20);
sumd = zeros(20,20);
for i=2:20
[clust(:,i), kCents{i}, sumd(1:i,i)] = kmeans(features_reduced2d{5},i,'emptyaction','singleton',...
        'replicate',5);
end
k_sse = sum(sumd, 1);
plot(2:20, k_sse(2:20)); title('K-Means Elbow Plot');
xlabel('k (num clusters)'); ylabel('Sum Squared Errors (Dist. from Centroids');

%% visualize cluster HR
hold on;
tvis = [1:length(actual_testHR),]';
plt_colors = distinguishable_colors(clusterNum);
for j = 1:clusterNum
    scatter(tvis(cluster_inds == j,:),actual_testHR(cluster_inds == j,:), 20, plt_colors(j,:), '.')
    title('Optimal Clustering');
end
