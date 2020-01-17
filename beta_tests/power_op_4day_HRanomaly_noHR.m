% Changelog: Added cross validation since v3
% v9: cross validation with each day as one fold (most sections do not 
% work because this code is specifically used for CV across all 4 days
% Look at v8 for code used in thesis paper using GCV and LCV

%This version attempts to predict HR rather than BR

%v2 fixing bugs with the delta target HR, making it predict actual HR
%instead of change in HR

clear all;
clc;


% LOAD DATA
tic
working_dir = pwd;

dayPostData = {
    [working_dir '\Data\y18m02d14\Post\DataFeatures.mat'] % Day 5
    [working_dir '\Data\y18m02d16\Post\DataFeatures.mat'] % Day 6
    [working_dir '\Data\y18m03d01\Post\DataFeatures.mat'] % Day 7
    [working_dir '\Data\y18m03d01_2\Post\DataFeatures.mat'] % Day 8
    };

% dayPostData = {
%     'C:\Users\starl\Documents\ARoS_Lab\Power_Op\PowerOptimization-master\Data\y18m02d14\Post\DataFeatures.mat' % Day 5
%     'C:\Users\starl\Documents\ARoS_Lab\Power_Op\PowerOptimization-master\Data\y18m02d16\Post\DataFeatures.mat' % Day 6
%     'C:\Users\starl\Documents\ARoS_Lab\Power_Op\PowerOptimization-master\Data\y18m03d01\Post\DataFeatures.mat' % Day 7
%     'C:\Users\starl\Documents\ARoS_Lab\Power_Op\PowerOptimization-master\Data\y18m03d01_2\Post\DataFeatures.mat' % Day 8
%     };

clearvars features target_BR target_HR sessionStarts sessionEnds

% Parameters
trainDayInd = [1,2,3];
testDayInd = 4;
clusterNum = 1; %number of clusters (activities)
clusterFeatureInds = 25:45; % 4:24->wrist, 25:45->ankle
% last 4 columns are dBR d2BR dHR d2HR defined  further below
featureGroupIDs = [1 2 3 4*ones(1,21) 5*ones(1,21) 6*ones(1,6)]; 
groupNames = {'Body Temp', 'Env. Temp', 'Humidity', 'Wrist accel', 'Ankle accel', 'EDA'};
currentSensor = [0.312 0.263, 0.263, 0.312, 3.191, 0.312];
current_weights = currentSensor/sum(currentSensor) * 10;
currentWrist = 4.126;
currentChest = 5.819;
currentTotal = currentWrist + currentChest;
numGroups = length(groupNames);
lambdas = exp(linspace(log(0.01), log(30), 300)); %group lasso parameter
increaseAmounts = [0 0.01 0.03 0.05 0.1 0.15 0.20]; %RMSE increase percentages for accuracy trade off analysis
FL = 0; % forecast length (s)
section_days = [1 2 3 4];
test_section_days = testDayInd;%[1 2 3 4];

% PARSE DATA

for dayInd = 1:numel(dayPostData)
    load(dayPostData{dayInd}); % loads vars into variables  t x y
    dayData = [x(:,[1,3:52]) x(:,2)];
    %moving average to smooth BR data
    %dayData(:, 53) = movmean(dayData(:, 53), [4 0]);
    %dayData(:, 2) = movmean(dayData(:, 2), [4 0]);
    %Calculate differences between adjacent values BR and HR
   % dBR = diff(dayData(:, 53)); dBR = [dBR(1) ; dBR];
    %d2BR = diff(dBR); d2BR = [d2BR(1) ; d2BR];
    dHR = diff(dayData(:, 52)); dHR = [dHR(1) ; dHR];
    d2HR = diff(dHR); d2HR = [d2HR(1) ; d2HR];
    %dayData = [dayData dHR d2HR];
    %find breaks in the session and save the start and end session indices
    sessionBreaks = find(diff(t) > 5);
    sessionStarts{dayInd} = [1 ; sessionBreaks + 1];
    sessionEnds{dayInd} = [sessionBreaks ; numel(t)];
    time{dayInd} = [];
    features{dayInd} = [];
    features_HR{dayInd} = [];
    target_BR{dayInd} = [];
    target_HR{dayInd} = [];
    target_HR{dayInd} = [];
    %Parse data according to session breaks as t vector indices
    %Target BR and HR should have elements shifted by forecast length
    for sessionStartsInd = 1:numel(sessionStarts{dayInd})
        time{dayInd} = [time{dayInd} ; t(sessionStarts{dayInd}(sessionStartsInd)+ ... 
            FL:sessionEnds{dayInd}(sessionStartsInd))];
        features{dayInd} = [features{dayInd} ; dayData(sessionStarts{dayInd}(sessionStartsInd)+ ... 
            FL:sessionEnds{dayInd}(sessionStartsInd), 1:51)];
        %features_HR{dayInd} = [features_HR{dayInd} ; dayData(sessionStarts{dayInd}(sessionStartsInd): ... 
           % sessionEnds{dayInd}(sessionStartsInd)-FL, 52:54)];
        target_HR{dayInd} = [target_HR{dayInd} ; dayData(sessionStarts{dayInd}(sessionStartsInd) ... 
            +FL:sessionEnds{dayInd}(sessionStartsInd), 52)];
    end
    %features{dayInd} = [features{dayInd} features_HR{dayInd}];
    % Fix nan in features
    for col=1:size(features{dayInd},2)
        nanx = isnan(features{dayInd}(:,col));
        t = 1:numel(features{dayInd}(:,col));
        features{dayInd}(nanx,col) = interp1(t(~nanx), x(~nanx), t(nanx));
    end
    %Calculate change in BR/HR over the forecast length
    %target_BR{dayInd} = target_BR{dayInd} - features{dayInd}(:, 53); % delta target
    %target_HR{dayInd} = target_HR{dayInd} - features_HR{dayInd}(:,1); % delta target
end

%% Clustering

disp('Started');
section_days = [1 2 3 4];
test_section_days = testDayInd;%[1 2 3 4];
clear optimal_lambda MSE_BR;
tic

for test_section_day = test_section_days
    %disp(test_section_day * 100);
    %train_section_days = section_days([1:section_day_ind-1 section_day_ind+1:end]);
    %test_section_days = section_days(section_day_ind);
    train_section_days = section_days(section_days ~= test_section_day);
    
    % CLUSTER
    %Normalize train section features using zscore
    [train_sections_features_Z, features_mu, features_sigma] = zscore(vertcat(features{train_section_days}));
    %dimensionality reduction toolbox found online
    %reducedDimData = compute_mapping(, 'LPP');
    [train_sections_features_Z_reduced, mapping] = compute_mapping(train_sections_features_Z(:, clusterFeatureInds), 'LPP', 3);
    [train_cluster_inds{test_section_day}, cluster_centroids{test_section_day}] = kmeans(train_sections_features_Z_reduced, clusterNum); % centroids not needed
    %Normalize train sections target BR using zscore
    [train_sections_targetHR_Z, targetHR_mu, targetHR_sigma] = zscore(vertcat(target_HR{train_section_days}));

    cluster_count = clusterNum;

    % sort clusters so that lower cluster number has more points
    clusterSize = [];
    for cluster = 1:cluster_count
        clusterSize(cluster) = sum(train_cluster_inds{test_section_day} == cluster);
    end
    [~, ind] = sort(clusterSize, 'descend');
    new_cluster_inds = [];
    new_centroids = [];
    for cluster = 1:cluster_count
        new_cluster_inds(train_cluster_inds{test_section_day} == ind(cluster)) = cluster;
        new_centroids(cluster, :) = cluster_centroids{test_section_day}(ind(cluster), :);
    end
    train_cluster_inds{test_section_day} = new_cluster_inds;
    cluster_centroids{test_section_day} = new_centroids;
    
    %Apply standardization, feature reduction and clustering on each day of data
    %based on test section day training days parameters 
    for section_day = section_days
        disp([num2str(test_section_day) ', ' num2str(section_day)]);
        %Normalize data: mean shifted and divide by std
        features_Z{section_day} = bsxfun(@rdivide, bsxfun(@minus, features{section_day}, features_mu), features_sigma);
        %out_of_sample part of dim reduction toolkit for applying
        %previously calculated dim reduction mapping using training
        %features on test day
        features_Z_reduced{section_day} = out_of_sample(features_Z{section_day}(:, clusterFeatureInds), mapping);
        % cluster_inds : clustering based on test_section_day (1st index) cluster centroids and applied to each day (2nd index)
        cluster_inds{test_section_day}{section_day} = kmeans(features_Z_reduced{section_day}, clusterNum, 'Start', cluster_centroids{test_section_day});

    end
end
% 
% save('cluster_vars', 'section_days', 'test_section_days', 'cluster_inds');
% toc

%% MAIN Group lasso and average NRMSE
% if clusterNum == 1
%     cluster_file = ['results\K_sqrtP\delHR_' num2str(FL) 's_nocluster.mat'];
% else
%     cluster_file = ['results\K_sqrtP\delHR_' num2str(FL) 's.mat'];
% end
% load(cluster_file, 'cluster_inds');
tic
cluster_count = clusterNum;
for test_section_day = testDayInd
    train_section_days = section_days(section_days ~= test_section_day);
    [train_sections_features_Z, features_mu, features_sigma] = zscore(vertcat(features{train_section_days}));
    [train_sections_targetHR_Z, targetHR_mu, targetHR_sigma] = zscore(vertcat(target_HR{train_section_days}));
    %Calculate mean HR of clusters and create new target HR vector of
    %differences from mean       
    for section_day = section_days
        targetHR_Z{section_day} = bsxfun(@rdivide, bsxfun(@minus, target_HR{section_day}, targetHR_mu), targetHR_sigma);
        features_Z{section_day} = bsxfun(@rdivide, bsxfun(@minus, features{section_day}, features_mu), features_sigma);
    end
    
    for cluster = 1:cluster_count
        %Extract features for train and test days for specific cluster
        for CV_test_day = train_section_days
            CV_train_days = train_section_days(train_section_days ~= CV_test_day);   
            %CV_train_features = vertcat(features{CV_train_days});
            CV_train_features_Z = vertcat(features_Z{CV_train_days});
            CV_train_features_Z = CV_train_features_Z(vertcat(cluster_inds{test_section_day}{CV_train_days}) == cluster, :);
            %CV_train_targetBR = vertcat(target_BR{CV_train_days});
            CV_train_targetHR_Z = vertcat(targetHR_Z{CV_train_days});
            CV_train_targetHR_Z = CV_train_targetHR_Z(vertcat(cluster_inds{test_section_day}{CV_train_days}) == cluster, :);
            CV_test_features = features{CV_test_day}(cluster_inds{test_section_day}{CV_test_day} == cluster, :);
            %CV_test_features_HR = features_HR{CV_test_day}(cluster_inds{test_section_day}{CV_test_day} == cluster, :);
            CV_test_features_Z = features_Z{CV_test_day}(cluster_inds{test_section_day}{CV_test_day} == cluster, :);
            CV_test_targetHR = target_HR{CV_test_day}(cluster_inds{test_section_day}{CV_test_day} == cluster, :);
            %CV_test_targetBR_Z = targetBR_Z{CV_test_day};
            
            %features_Z_clustered{section_day_2} = features_Z{section_day_2}(cluster_inds{section_day_2} == cluster, :);

            % orth CV train and CV test features (for group lasso)
            CV_train_features_Z_orth = [];
            CV_test_features_Z_orth = [];
            %Singular value decomposition on on features matrix 
            for g = 1:max(featureGroupIDs)
                %disp(g);
                [CV_U, CV_S, CV_V] = svd(CV_train_features_Z(:, featureGroupIDs == g), 'econ'); % A = U * S * V' --- U = A * V * inv(S)
                if isequal(CV_S, 0)
                    disp('bad S');
                    CV_S = Inf;
                end
                CV_train_features_Z_orth(:, featureGroupIDs == g) = CV_U;
                %project test features onto orthonormal basis of train
                %features
                CV_test_features_Z_orth(:, featureGroupIDs == g) = CV_test_features_Z(:, featureGroupIDs == g) * CV_V * inv(CV_S);
            end
            
            %Group lasso
            CV_b_grp_HR = [];
            for lambdaInd = 1:length(lambdas)
                % Train group lasso (returns regression coefficients)
                CV_b_grp_HR(:, lambdaInd) = grplassoShooting(CV_train_features_Z_orth, CV_train_targetHR_Z, featureGroupIDs, lambdas(lambdaInd), 2e4, 1e-10, false);

                % Test group lasso on BR
                CV_predicted_HR = (CV_test_features_Z_orth * CV_b_grp_HR(:, lambdaInd)) * targetHR_sigma + targetHR_mu; % + CV_test_features(:,52);
                %smooth_CV_predicted_BR = movmean(CV_predicted_BR, [4 0]);
                CV_actualHR = CV_test_targetHR; %+ CV_test_features(:, 52);
                err = CV_predicted_HR - CV_actualHR;
                %err = smooth_CV_predicted_BR - CV_actualBR;
                %Mean square error per cv test day per lambda value
                CV_MSE_HR{CV_test_day}(lambdaInd) = rms(err) ^ 2;
                CV_RMSE_HR{CV_test_day}(lambdaInd) = rms(err);
                CV_NRMSE_HR{CV_test_day}(lambdaInd) = rms(err)/range(CV_actualHR);
            end
        end
        %Calculate average MSE across CV train days for each lambda and
        %find lambda that minimizes MSE
        avg_CV_MSE_HR{test_section_day}{cluster} = mean(vertcat(CV_MSE_HR{train_section_days}), 1); % average across CV train days
        avg_CV_RMSE_HR{test_section_day}{cluster} = mean(vertcat(CV_RMSE_HR{train_section_days}), 1); % average across CV train days
        avg_CV_NRMSE_HR{test_section_day}{cluster} = mean(vertcat(CV_NRMSE_HR{train_section_days}), 1);
        [~, optimal_lambda{test_section_day}{cluster}] = min(avg_CV_MSE_HR{test_section_day}{cluster});
        
        % test model
        train_features_Z = vertcat(features_Z{train_section_days});
        train_features_Z = train_features_Z(vertcat(cluster_inds{test_section_day}{train_section_days}) == cluster, :);
        train_targetHR_Z = vertcat(targetHR_Z{train_section_days});
        train_targetHR_Z = train_targetHR_Z(vertcat(cluster_inds{test_section_day}{train_section_days}) == cluster, :);
        test_features = features{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
        %test_features_HR = features_HR{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
        test_targetHR = target_HR{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
        test_features_Z = features_Z{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);

        % orthonormal train and test features (for group lasso)
        train_features_Z_orth = [];
        test_features_Z_orth = [];
        for g = 1:max(featureGroupIDs)
            %disp(g);
            [CV_U, CV_S, CV_V] = svd(train_features_Z(:, featureGroupIDs == g), 'econ'); % A = U * S * V' --- U = A * V * inv(S)
            if isequal(CV_S, 0)
                disp('bad S 2');
                CV_S = Inf; 
            end
            train_features_Z_orth(:, featureGroupIDs == g) = CV_U;
            test_features_Z_orth(:, featureGroupIDs == g) = test_features_Z(:, featureGroupIDs == g) * CV_V * inv(CV_S);
        end
        %b_grp_BR = [];
        for lambdaInd = 1:length(lambdas)
            % Train group lasso
            b_grp_HR = grplassoShooting(train_features_Z_orth, train_targetHR_Z, featureGroupIDs, lambdas(lambdaInd), 2e4, 1e-10, false);
            
            % Test group lasso on BR
            predicted_HR{test_section_day}{cluster}(:,lambdaInd) = (test_features_Z_orth * b_grp_HR) * targetHR_sigma + targetHR_mu;% + test_features(:,52);
            %smooth_CV_predicted_BR = movmean(predicted_BR, [4 0]);
            actualHR = test_targetHR; % + test_features(:,52);
            err = predicted_HR{test_section_day}{cluster}(:,lambdaInd) - actualHR;
            %err = smooth_CV_predicted_BR - CV_actualBR;
            MSE_HR{test_section_day}{cluster}(lambdaInd) = rms(err) ^ 2;
            RMSE_HR{test_section_day}{cluster}(lambdaInd) = rms(err);
            NRMSE_HR{test_section_day}{cluster}(lambdaInd) = sqrt(MSE_HR{test_section_day}{cluster}(lambdaInd)) / range(actualHR);
            numFeatures_HR{test_section_day}{cluster}(lambdaInd) = sum(b_grp_HR ~= 0);
            %sensorGroupsUsed{test_section_day}{cluster}(lambdaInd) =
            %unique(featureGroupIDs(b_grp_BR ~= 0)); 
            sensorGroupsUsed{test_section_day}{cluster}(lambdaInd, 1:max(featureGroupIDs)) = false;
            sensorGroupsUsed{test_section_day}{cluster}(lambdaInd, unique(featureGroupIDs(b_grp_HR ~= 0))) = true;
            numSensors_HR{test_section_day}{cluster}(lambdaInd) = length(unique(featureGroupIDs(b_grp_HR ~= 0)));
        end
        
    end
    
    % average across clusters (with each cluster's optimal lambda)
    avg_MSE_HR{test_section_day} = 0;
    avg_RMSE_HR{test_section_day} = 0;
    for cluster = 1:cluster_count
        avg_MSE_HR{test_section_day} = avg_MSE_HR{test_section_day} + MSE_HR{test_section_day}{cluster}(optimal_lambda{test_section_day}{cluster}) * sum(cluster_inds{test_section_day}{test_section_day} == cluster);
        avg_RMSE_HR{test_section_day} = avg_RMSE_HR{test_section_day} + RMSE_HR{test_section_day}{cluster}(optimal_lambda{test_section_day}{cluster}) * sum(cluster_inds{test_section_day}{test_section_day} == cluster);
        avg_RMSE_HR_cluster(cluster) = RMSE_HR{test_section_day}{cluster}(optimal_lambda{test_section_day}{cluster});  
    end
    avg_MSE_HR{test_section_day} = avg_MSE_HR{test_section_day} / length(cluster_inds{test_section_day}{test_section_day});
    avg_RMSE_HR{test_section_day} = avg_RMSE_HR{test_section_day} / length(cluster_inds{test_section_day}{test_section_day});
    avg_NRMSE_HR{test_section_day} = avg_RMSE_HR{test_section_day}/range(target_HR{test_section_day});
    
end

disp('Finished');
toc


%% Find some average error values

actual_testHR = target_HR{testDayInd};% + features{testDayInd}(:,52);
avg_avg_MSE_HR = mean(vertcat(avg_MSE_HR{test_section_days})); % average BR MSE across days/folds
avg_avg_RMSE_HR = mean(vertcat(avg_RMSE_HR{test_section_days})); % average BR MSE across days/folds
avg_avg_NRMSE_HR = avg_avg_RMSE_HR/range(actual_testHR);
minLambda_predicted_HR = [];


for cluster = 1:cluster_count
    avg_RMSE_HR_cluster(cluster,:) = RMSE_HR{testDayInd}{cluster}(optimal_lambda{testDayInd}{cluster}); 
    minLambda_predicted_HR(cluster_inds{testDayInd}{testDayInd} == cluster) = ...
        predicted_HR{testDayInd}{cluster}(:,optimal_lambda{testDayInd}{cluster});
    err_final_rms_cluster(cluster) = rms(minLambda_predicted_HR(cluster_inds{testDayInd}{testDayInd} == cluster)' ...
        - actual_testHR(cluster_inds{testDayInd}{testDayInd} == cluster));
    err_meanDummy(cluster_inds{test_section_day}{test_section_day} == cluster, :) ... 
       = actual_testHR(cluster_inds{test_section_day}{test_section_day} == cluster, :) - ...
       mean(actual_testHR(cluster_inds{test_section_day}{test_section_day} == cluster));
    c_num{cluster,:} = sum(cluster_inds{testDayInd}{testDayInd} == cluster);
    c_p{cluster,:} = c_num{cluster}/length(cluster_inds{testDayInd}{testDayInd});
end

err_final = minLambda_predicted_HR' - actual_testHR;
results.err = err_final;
results.rms = rms(err_final);
results.nrms = results.rms/range(actual_testHR);
results.rmsr = rms(err_final./actual_testHR);
results.mae = sum(abs(err_final))/length(err_final);
results.mape = sum(abs(err_final./actual_testHR))/length(err_final);
results.r2 = 1 - sum(err_final.^2)/sum((actual_testHR-mean(actual_testHR)).^2);
res_cp.rms = rms(err_meanDummy);
res_cp.nrms = res_cp.rms/range(actual_testHR);
res_cp.rmsr = rms(err_meanDummy./actual_testHR);
res_cp.mae = sum(abs(err_meanDummy))/length(actual_testHR);
res_cp.mape = sum(abs(err_meanDummy./actual_testHR))/length(actual_testHR);
res_cp.r2 = 1 - sum(err_meanDummy.^2)/sum((actual_testHR-mean(actual_testHR)).^2);



%% Get list of features used per test day and cluster and optimal features
%used for lambda with min RMSE

clear optimalFeaturesUsed featuresUsed optimalsensorGroupsUsed;
groupInds = 1:max(featureGroupIDs);
currentReduction_avg = 0;
for test_section_day = test_section_days
    for cluster = 1:cluster_count
        %initialize
        featuresUsed{test_section_day}{cluster} = false(length(lambdas), length(featureGroupIDs));
        for lambdaInd = 1:length(lambdas)
            for groupInd = groupInds(sensorGroupsUsed{test_section_day}{cluster}(lambdaInd, :))
                enabledFeatures = featureGroupIDs == groupInd;
                featuresUsed{test_section_day}{cluster}(lambdaInd, enabledFeatures) = true;
            end
        end
        %features used with optimal lambda
        lambdaInd = optimal_lambda{test_section_day}{cluster};
        optimalFeaturesUsed{test_section_day}{cluster} = featuresUsed{test_section_day}{cluster}(lambdaInd, :);
        optimalsensorGroupsUsed{test_section_day}{cluster} = sensorGroupsUsed{test_section_day}{cluster}(lambdaInd, :);
        currentReduction{test_section_day}(1, cluster) = sum(currentSensor .* not(optimalsensorGroupsUsed{test_section_day}{cluster}));
        currentReduction{test_section_day}(2, cluster) = currentReduction{test_section_day}(1, cluster)/currentTotal;
        currentReduction_avg(1, :) = currentReduction_avg + currentReduction{test_section_day}(1, cluster)*c_p{cluster};
    end
    currentReduction_avg(2, :) = currentReduction_avg(1, :)/currentTotal;
end
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

for i = 2:2
    if i == 1
       svm_hr;
    else
       svm_allFeats_hr;
    end
    svm_actual_HR_full = vertcat(svm_actual_HR{testDayInd}{1:cluster_count});
    svm_predicted_HR_full = vertcat(svm_predicted_HR{testDayInd}{1:cluster_count});
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
return;

%% Plot feature/sensor disappearance
figure
clf

for cluster = 1:clusterNum
    h = subplot(1, clusterNum, cluster);
    feature_exist = featuresUsed{testDayInd}{cluster}';
    feature_exist = [ones(size(feature_exist, 1), 1) feature_exist];
    feature_exist(:, end) = 0;
    feature_lost_table = diff(feature_exist, 1, 2);
    feature_lost_table = feature_lost_table([1 2 3 4 25 46], :);
    [feature_lost, lambda_lost] = find(feature_lost_table == -1);
    feature_lost = flipud(feature_lost);
    lambda_lost = flipud(lambda_lost);
    [feature_lost_latest, idxa, idxb] = unique(feature_lost, 'stable');
    lambda_lost_latest = lambda_lost(idxa);
    feature_lost_latest = flipud(feature_lost_latest);
    lambda_lost_latest = flipud(lambda_lost_latest);
%     [feature_lost_latest, sort_idx] = sort(feature_lost_latest);
%     lambda_lost_latest = lambda_lost_latest(sort_idx)
    sensor_lost = groupNames(feature_lost_latest);
  
    hold on;
    minLambdaInd = optimal_lambda{testDayInd}{cluster};
%     bar(find(lambda_lost_latest < minLambdaInd), lambdas(lambda_lost_latest(lambda_lost_latest < minLambdaInd)), 'FaceColor', [1 0.5 0.3]);
%     bar(find(lambda_lost_latest >= minLambdaInd), lambdas(lambda_lost_latest(lambda_lost_latest >= minLambdaInd)), 'FaceColor', [0.3 1 0.5]);
    bar(1:sum(lambda_lost_latest < minLambdaInd), lambdas(lambda_lost_latest(lambda_lost_latest < minLambdaInd)), 'FaceColor', [1 0.5 0.3]);
    bar((numGroups+1)-sum(lambda_lost_latest >= minLambdaInd):numGroups, lambdas(lambda_lost_latest(lambda_lost_latest >= minLambdaInd)), 'FaceColor', [0.3 1 0.5]);
    plot([0 numGroups+1], lambdas(minLambdaInd) * ones(2, 1), 'k--');

    set(h, 'yscale', 'log')
    set(h, 'FontSize', 15);
    set(h, 'XLim', [0 numGroups+1]);
    set(h, 'YLim', [lambdas(1) / 1.1 lambdas(end)]);
    grid on;
    ylabel('\lambda');
    %title(['Group lasso lambda at which sensors are excluded - Cluster = ' num2str(cluster)]);
    title(['Cluster = ' num2str(cluster)]);
    xticks(1:length(feature_lost_latest));
    xtickangle(45);
    xticklabels(sensor_lost);

end


%% Plot RMSE and numSensors
figure;
clf;

for test_section_day = 3
    for cluster = 1:cluster_count
%         lambdaInd = optimal_lambda{test_section_day}{cluster};
        %subplot(length(test_section_days), cluster_count, (test_section_day-1)*cluster_count + cluster);
        subplot(1, cluster_count, cluster);
        yyaxis left
        ylim([-inf max(max(sqrt(avg_CV_MSE_HR{test_section_day}{cluster})),max(sqrt(MSE_HR{test_section_day}{cluster})))]);
        hold on;
        plot(lambdas, sqrt(avg_CV_MSE_HR{test_section_day}{cluster}), 'r');
        plot(lambdas, sqrt(MSE_HR{test_section_day}{cluster}), 'b-');
        xlabel('\lambda')
        ylabel('RMSE');
        xticks([0.01 1 10 30])
        yyaxis right
        ylim([0 numGroups]) 
        yticks([0 numGroups])
        plot(lambdas, numSensors_HR{test_section_day}{cluster}, 'k--');
        ylabel('Number of Sensors');
        rmsPlotTitle = ['Cluster = ' num2str(cluster)];
        title(rmsPlotTitle);
        ax = gca;
        ax.YAxis(2).Color = 'black';
        
        set(gca, 'xscale', 'log')
        set (gca, 'fontsize', 10)
        set(gca, 'XLim', [lambdas(1) / 1.1 lambdas(end)]);
    end
end

rmsLgd = legend('CV RMSE', 'Test RMSE', 'Num Sensors');
rmsLgd.FontSize = 8;


%% Plot NRMSE and numSensors
figure(2);
clf;

for test_section_day = testDayInd
    for cluster = 1:cluster_count
        subplot(1, cluster_count, cluster);
        yyaxis left
        ylim([-inf max(max(avg_CV_NRMSE_HR{test_section_day}{cluster},max(NRMSE_HR{test_section_day}{cluster})))]);
        hold on;
        plot(lambdas, (avg_CV_NRMSE_HR{test_section_day}{cluster}), 'b');
        plot(lambdas, (NRMSE_HR{test_section_day}{cluster}), 'b--');
        xlabel('\lambda')
        ylabel('RMSE');
        xticks([0.01 1 10 40])
        yyaxis right
        ylim([0 numGroups]) 
        yticks([0 numGroups])
        plot(lambdas, numSensors_HR{test_section_day}{cluster});
        ylabel('Num Sensors');
        
        set(gca, 'xscale', 'log')
        set (gca, 'fontsize', 12)
        set(gca, 'XLim', [lambdas(1) / 1.1 lambdas(end)]);
        
    end
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
% plot(lambdas, avg_CV_MSE_HR{test_section_day}{cluster}, 'b');
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
%% Visualize test data clusters 
figure
clf;
hold on;
cluster_inds_train = vertcat(cluster_inds{testDayInd}{trainDayInd});
features_Z_reduced_train = vertcat(features_Z_reduced{trainDayInd});
for cluster = 1:clusterNum
    plotData = features_Z_reduced_train(cluster_inds_train == cluster, :);
    h = plot3(plotData(:,1), plotData(:,2), plotData(:,3), '.', 'MarkerSize', 6);
    %color = get(h, 'Color');
    %plotData = testAccelDataReduced(testClusterInd == cluster, :);
    %plot3(plotData(:,1), plotData(:,2), plotData(:,3), '+', 'MarkerSize', 6, 'Color', color);
end
%plot3(testAccelDataReduced(:,1), testAccelDataReduced(:,2), testAccelDataReduced(:,3), '.', 'MarkerSize', 4);
%legend(cellfun(@num2str, num2cell(reshape(repmat(1:clusterCount, 2, 1)), clusterCount*2, 1), 'uniform', 0));
%legend({'1', '1', '2', '2', '3', '3', '4', '4'})
legend(['1 (' num2str(sum(cluster_inds_train == 1)) ')'],...
['2 (' num2str(sum(cluster_inds_train == 2)) ')'],...
['3 (' num2str(sum(cluster_inds_train == 3)) ')'],...
['4 (' num2str(sum(cluster_inds_train == 4)) ')']);
% objhl = findobj(objh, 'type', 'line'); %// objects of legend of type line
% set(objhl, 'Markersize', 10);
% objtl = findobj(objh, 'type', 'text'); %// objects of legend of type line
% set(objtl, 'Fontsize', 14);
%title('Clustering Visualization');
axis equal;
set(gca, 'FontSize', 12);
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
set(gca,'ZTickLabel',[]);

% %%
% % figHandles = findobj('Type', 'figure');
% % for fhInd = 1:numel(figHandles)
% %     figHandles(fhInd).CurrentAxes.DataAspectRatioMode = 'manual';
% %     figHandles(fhInd).CurrentAxes.DataAspectRatio = [1 1 1];
% % end
% 
%% Visualize testing cluster over time 
% if clusterNum == 1
%     cluster_file = ['results\K_sqrtP\delHR_' num2str(FL) 's_nocluster.mat'];
% else
%     cluster_file = ['results\K_sqrtP\delHR_' num2str(FL) 's.mat'];
% end
% load(cluster_file, 'features_Z_reduced');
% clear new_clusters
new_clusters = zeros(length(cluster_inds{testDayInd}{testDayInd}),1);
new_clusters(cluster_inds{testDayInd}{testDayInd} == 1, :) = 2;
new_clusters(cluster_inds{testDayInd}{testDayInd} == 2, :) = 3;
new_clusters(cluster_inds{testDayInd}{testDayInd} == 3, :) = 1;
new_clusters(cluster_inds{testDayInd}{testDayInd} == 4, :) = 4;
modefun = @(x) mode(x(:));
cluster_mode = nlfilter(cluster_inds{testDayInd}{testDayInd}, [20 1], modefun);
smoothHRpredict = movmean(minLambda_predicted_HR, [0 4]);
figure;
time = 1:length(cluster_mode);
subplot(3,1,1); hold on
for ii = 1:1
h1= plot(time/60, features_Z_reduced{testDayInd}(:, ii)); 
%legend('IMU Feature')
end
hold off
ylabel('IMU');
set(gca, 'fontsize', 10)
hold off
h2 = subplot(3,1,2); plot(time/60, cluster_mode, '.', 'MarkerSize', 12); 
ylabel('Cluster Number');
set(gca, 'fontsize', 10)
%legend('Cluster');
subplot(3,1,3); hold on
h3 = plot(time/60, actual_testHR, time/60, smoothHRpredict); 
legend('Measured', 'Predicted');
hold off
ylabel('HR (bpm)')
xlabel('t (mins)')
set(gca, 'fontsize', 10)
%lgd = legend([h1, h2, h3, h4],{'IMU Feature', 'Cluster', 'Measure HR', 'Predicted HR'});
%set(gca, 'YTicks', [0 1 2 3 4 5])


%% Find the lambda indices given by the increase allowed RMSE amount vector

for test_section_day = test_section_days
    for cluster = 1:cluster_count
        for increaseAmountInd = 1:length(increaseAmounts)
            increaseAmount = increaseAmounts(increaseAmountInd);
            %Find index of lambda with RMSE closest to increased RMSE 
            [~, increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd)] = min(...
                abs(sqrt(avg_CV_MSE_HR{test_section_day}{cluster}(optimal_lambda{test_section_day}{cluster}+1:end)) - ...
                (1+increaseAmount)*sqrt(avg_CV_MSE_HR{test_section_day}{cluster}(optimal_lambda{test_section_day}{cluster})))...
                );
            increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd) = ...
                increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd) + optimal_lambda{test_section_day}{cluster};
            %         [~, increaseHRAmountLambdaInd{cluster}(increaseAmountInd)] = min(abs(CV_avg_RMSE_HR{cluster}(minHRCVLambdaInd+1:end) - (1+increaseAmount)*min(CV_avg_RMSE_HR{cluster}(minHRCVLambdaInd+1:end))));
            %         increaseHRAmountLambdaInd{cluster}(increaseAmountInd) = increaseHRAmountLambdaInd{cluster}(increaseAmountInd) + minHRCVLambdaInd;
        end
        increaseHRAmountLambdaInd{test_section_day}{cluster}(1) = ...
                increaseHRAmountLambdaInd{test_section_day}{cluster}(1) - 1;
     end
end

%% Calculate average RMSE, number sensors, and cluster counts using the new
%  increased allowable RMSE percentage lambda index (lambdaInc)
RMSE_increase = 0.03;
disp('RMSE for days');
RMSE_results = [];
avg_MSE_HR_it = [];
for test_section_day = section_days
    %disp(['Day ' num2str(test_section_day)]);
    %MSE_HR{test_section_day}{cluster}(lambdaInd)

    %avg_MSE_HR_it(test_section_day) = 0;
    avg_MSE_HR_it = 0;
    avg_numSensors_HR = 1;
    for cluster = 1:cluster_count
        %disp(['Cluster ' num2str(cluster)]);
%         lambdaInd = optimal_lambda{test_section_day}{cluster};
        increaseAmountInd = find(increaseAmounts == RMSE_increase);
        lambdaInd = increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd);
        RMSE_results(test_section_day * 3 - 2, cluster) = sqrt(MSE_HR{test_section_day}{cluster}(lambdaInd));
        RMSE_results(test_section_day * 3 - 1, cluster) = numSensors_HR{test_section_day}{cluster}(lambdaInd);
        RMSE_results(test_section_day * 3    , cluster) = sum(cluster_inds{test_section_day}{test_section_day} == cluster);
        
        %avg_MSE_HR_it(test_section_day) = avg_MSE_HR_it(test_section_day) + MSE_HR{test_section_day}{cluster}(lambdaInd) * sum(cluster_inds{test_section_day}{test_section_day} == cluster);
        avg_MSE_HR_it = avg_MSE_HR_it + MSE_HR{test_section_day}{cluster}(lambdaInd) * sum(cluster_inds{test_section_day}{test_section_day} == cluster);
        avg_numSensors_HR = avg_numSensors_HR + numSensors_HR{test_section_day}{cluster}(lambdaInd) * sum(cluster_inds{test_section_day}{test_section_day} == cluster);
        %disp([num2str(sum(cluster_inds{test_section_day} == cluster)) ' ' num2str(MSE_HR{test_section_day}{cluster}(lambdaInd))]);
    end
    %avg_MSE_HR_it(test_section_day) = avg_MSE_HR_it(test_section_day) / length(cluster_inds{test_section_day}{test_section_day});
    avg_MSE_HR_it = avg_MSE_HR_it / length(cluster_inds{test_section_day}{test_section_day});
    avg_numSensors_HR = avg_numSensors_HR / length(cluster_inds{test_section_day}{test_section_day});
    avg_clusterNum = length(cluster_inds{test_section_day}{test_section_day}) / cluster_count;

    RMSE_results(test_section_day * 3 - 2, cluster_count+1) = sqrt(avg_MSE_HR_it);
    RMSE_results(test_section_day * 3 - 1, cluster_count+1) = avg_numSensors_HR;
    RMSE_results(test_section_day * 3    , cluster_count+1) = avg_clusterNum;

end
avg_RMSE_HR_it = sqrt(avg_MSE_HR_it);
avg_RMSE_HR_it = round(avg_RMSE_HR_it' * 1000) / 1000;
RMSE_results = round(RMSE_results * 1000) / 1000;
%disp(avg_avg_MSE_HR);

%% Sensors used for each day using new allowable increased RMSE value 
clear current_reduction_new;
for test_section_day = test_section_days
    for increaseInd = 1:length(increaseAmounts)
        sensors_used_results{increaseInd} = false(max(featureGroupIDs), cluster_count);
        for cluster = 1:cluster_count
    %         lambdaInd = optimal_lambda{test_section_day}{cluster};
            %increaseAmountInd = find(increaseAmounts == RMSE_increase);
            increaseAmountInd = increaseInd;
            lambdaInd = increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd);

            %sensors_used_results(sensorGroupsUsed{test_section_day}{cluster}(lambdaInd), (test_section_day-1) * 4 + cluster) = true;
            sensors_used_results{increaseInd}(:, cluster) = sensorGroupsUsed{test_section_day}{cluster}(lambdaInd, :)';
            %unique(featureGroupIDs(b_grp_HR ~= 0))
        end
        current_reduction_new(increaseInd, 1:cluster_count) = currentSensor * not(sensors_used_results{increaseInd});
        current_reduction_new(increaseInd, cluster_count + 1) = 0;
        for cluster = 1:cluster_count
            current_reduction_new(increaseInd, cluster_count + 1) = current_reduction_new(increaseInd, cluster_count + 1) + current_reduction_new(increaseInd, cluster)*c_p{cluster};
        end
    end
end

% sensors_used_results;
% 

%% Try different dimensionality reduction techniques 

% close all;
% clc;
% 
% techniqueNames = {'PCA', 'LDA', 'MDS', 'ProbPCA', 'FactorAnalysis', 'GPLVM', ...
%     'Sammon', 'Isomap', 'LandmarkIsomap', 'LLE', 'Laplacian', 'HessianLLE', ...
%     'LTSA', 'MVU', 'CCA', 'LandmarkMVU', 'FastMVU', 'DiffusionMaps', ...
%     'KernelPCA', 'GDA', 'SNE', 'SymSNE', 'tSNE', 'LPP', 'NPE', 'LLTSA', ...
%     'SPE', 'Autoencoder', 'LLC', 'ManifoldChart', 'CFA', 'NCA', 'MCML', 'LMNN'};
techniqueNames = {'tSNE'}
% clusterFeatureInds = 5:46; % 5:46 5:25
% skip 6:9 12:17 21:23 29:31 33:34
for techniqueNamesInd = [1]
%     disp([num2str(techniqueNamesInd) '/' num2str(numel(techniqueNames)) ' - ' techniqueNames{techniqueNamesInd}]);
%     [accelDataReduced3{testDayInd}, mapping] = compute_mapping(features_Z{testDayInd}(:, clusterFeatureInds), techniqueNames{techniqueNamesInd}, 3, 21, 30);
%     [accelDataReduced2{testDayInd}, mapping] = compute_mapping(features_Z{testDayInd}(:, clusterFeatureInds), techniqueNames{techniqueNamesInd}, 2, train_sections_features_Z_reduced, 30);
%     epsilon = 0.009;
%     MinPts = 50;
%     %DBSCAN clustering function found online
%     trainClusterInd{testDayInd} = DBSCAN(accelDataReduced{testDayInd}, epsilon, MinPts);
%     clusterCount{testDayInd} = max(trainClusterInd{testDayInd});
    [train_sections_features_Z_tsne, mapping_tsne] = compute_mapping([train_sections_features_Z(:, clusterFeatureInds); features_Z{testDayInd}(:, clusterFeatureInds)], 'tSNE', 3);
    accelDataReduced3{testDayInd} = train_section_features_Z_tsne(size(train_sections_features_Z(:, clusterFeatureInds), 1)+1:end,:)
    train_section_features_Z_tsne = train_section_features_Z_tsne(1:size(train_sections_features_Z(:, clusterFeatureInds)), :)
    [train_cluster_inds_tsne, cluster_centroids_tsne] = kmeans(train_sections_features_Z_tsne, clusterNum);
    trainclusterSize = [];
    for cluster = 1:cluster_count
        trainclusterSize(cluster) = sum(trainClusterInd_tsne == cluster);
    end
    [~, ind] = sort(trainclusterSize, 'descend');
    new_cluster_inds = [];
    for cluster = 1:cluster_count
        new_cluster_inds(trainClusterInd_tsne == ind(cluster)) = cluster;
    end
    trainClusterInd_tsne = new_cluster_inds;
    accelDataReduced3{testDayInd}
    clusterInd_tsne = kmeans(accelDataReduced3{testDayInd}, 4, 'Start', cluster_centroids_tsne);
    clusterCount = 4;
    figure
    hold on;
    for cluster = 1:clusterCount
%         plotData = accelDataReduced3{testDayInd}(trainClusterInd{testDayInd} == cluster, :);
        plotData = accelDataReduced3{testDayInd}(clusterInd_tsne == cluster, :);
        plot3(plotData(:,1), plotData(:,2), plotData(:,3), '.', 'MarkerSize', 6);
%          plot(plotData(:,1), plotData(:,2), '.', 'MarkerSize', 6);
    end
    title(techniqueNames{techniqueNamesInd});
    legend(['none' cellfun(@num2str, num2cell(1:clusterCount), 'uniform', 0)]);
%     saveas(gcf,['reduced_' num2str(techniqueNamesInd) '.png']);
end

