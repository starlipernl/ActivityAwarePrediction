clear all;
clc;

% LOAD DATA
% enter your data directories here
dayPostData = {
    'C:\Users\starl\Documents\ARoS_Lab\Power_Op\PowerOptimization-master\Data\y18m02d14\Post\DataFeatures.mat' % Day 5
    'C:\Users\starl\Documents\ARoS_Lab\Power_Op\PowerOptimization-master\Data\y18m02d16\Post\DataFeatures.mat' % Day 6
    'C:\Users\starl\Documents\ARoS_Lab\Power_Op\PowerOptimization-master\Data\y18m03d01\Post\DataFeatures.mat' % Day 7
    'C:\Users\starl\Documents\ARoS_Lab\Power_Op\PowerOptimization-master\Data\y18m03d01_2\Post\DataFeatures.mat' % Day 8
    };

clearvars features target_BR target_HR sessionStarts sessionEnds


% 1 = airbeam
% 2 = bioharness
% 3 = e4
% 4 = sensortag
%featureGroup = [3 2 1 1 3*ones(1, 21) 4*ones(1,21) 3*ones(1,6)]; % per device

% Parameters
trainDayInd = 1:3;
testDayInd = 3;
clusterNum = 4; %number of clusters (activities)
clusterFeatureInds = 25:45; % 4:24->wrist, 25:45->ankle
% last 4 columns are dBR d2BR dHR d2HR defined  further below
featureGroupIDs = [1 2 3 4*ones(1,21) 5*ones(1,21) 6*ones(1,6) 7*ones(1,3)]; 
groupNames = {'Body Temp', 'Env. Temp', 'Humidity', 'Wrist accel', 'Ankle accel', 'EDA', 'Heart Rate'};
currentSensor = [0 0.263, 0.263, 0.312, 0.312, 0, 8.215];
currentIdle = 3.793;
currentFull = 12.341;
numGroups = length(groupNames);
lambdas = exp(linspace(log(0.01), log(30), 300)); %group lasso parameter
increaseAmounts = [0.01 0.03 0.05 0.1 0.15 0.2]; %RMSE increase percentages for accuracy trade off analysis
FL = 30; % forecast length (s)

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
    dayData = [dayData dHR d2HR];
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
        time{dayInd} = [time{dayInd} ; t(sessionStarts{dayInd}(sessionStartsInd)+FL:sessionEnds{dayInd}(sessionStartsInd))];
        features{dayInd} = [features{dayInd} ; dayData(sessionStarts{dayInd}(sessionStartsInd)+FL:sessionEnds{dayInd}(sessionStartsInd), 1:51)];
        features_HR{dayInd} = [features_HR{dayInd} ; dayData(sessionStarts{dayInd}(sessionStartsInd):sessionEnds{dayInd}(sessionStartsInd)-FL, 52:54)];
       % target_BR{dayInd} = [target_BR{dayInd} ; dayData(sessionStarts{dayInd}(sessionStartsInd)+FL:sessionEnds{dayInd}(sessionStartsInd), 53)];
        target_HR{dayInd} = [target_HR{dayInd} ; dayData(sessionStarts{dayInd}(sessionStartsInd)+FL:sessionEnds{dayInd}(sessionStartsInd), 52)];
    end
    features{dayInd} = [features{dayInd} features_HR{dayInd}];
    % Fix nan in features
    for col=1:size(features{dayInd},2)
        nanx = isnan(features{dayInd}(:,col));
        t = 1:numel(features{dayInd}(:,col));
        features{dayInd}(nanx,col) = interp1(t(~nanx), x(~nanx), t(nanx));
    end
    %Calculate change in BR/HR over the forecast length
    %target_BR{dayInd} = target_BR{dayInd} - features{dayInd}(:, 53); % delta target
    target_HR{dayInd} = target_HR{dayInd}; % - features_HR{dayInd}(:,1); % delta target
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
        new_cluster_inds = cluster_inds;
        for ii = 16:length(cluster_inds{test_section_day}{section_day})-16
            temp_count = sum(new_cluster_inds{test_section_day}{section_day}(ii-15:ii+15) == new_cluster_inds{test_section_day}{section_day}(ii));
            if temp_count < 16 
                new_cluster_inds{test_section_day}{section_day}(ii) = new_cluster_inds{test_section_day}{section_day}(ii-1);
            end
        end
        cluster_inds = new_cluster_inds;
        %normalize BR data

    end
end

save('cluster_vars', 'section_days', 'test_section_days', 'cluster_inds');
toc

%% MAIN Group lasso and average NRMSE
load('cluster_vars');
tic
cluster_count = clusterNum;
for test_section_day = testDayInd
    train_section_days = section_days(section_days ~= test_section_day);
    clear optimal_lambda MSE_BR;
    % Filter activity transitions:
    for section_day = section_days
        cluster_trans = [0; diff(cluster_inds{test_section_day}{section_day})];
        for i = find(cluster_trans)
            cluster_inds{test_section_day}{section_day}(i:i+25) = [];
            features{section_day}(i:i+25, :) = [];
            target_HR{section_day}(i:i+25, :) = [];
            features_HR{section_day}(i:i+25, :) = [];
        end    
    end
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
            CV_test_features_HR = features_HR{CV_test_day}(cluster_inds{test_section_day}{CV_test_day} == cluster, :);
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
                CV_b_grp_HR(:, lambdaInd) = grplassoShooting(CV_train_features_Z_orth, CV_train_targetHR_Z, featureGroupIDs, lambdas(lambdaInd), 1e4, 1e-10, false);

                % Test group lasso on BR
                CV_predicted_HR = (CV_test_features_Z_orth * CV_b_grp_HR(:, lambdaInd)) * targetHR_sigma + targetHR_mu; % + CV_test_features(:,52);
                %smooth_CV_predicted_BR = movmean(CV_predicted_BR, [4 0]);
                CV_actualHR = CV_test_targetHR;% + CV_test_features(:, 52);
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
        test_features_HR = features_HR{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
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
            b_grp_HR = grplassoShooting(train_features_Z_orth, train_targetHR_Z, featureGroupIDs, lambdas(lambdaInd), 1e4, 1e-10, false);
            
            % Test group lasso on BR
            predicted_HR{test_section_day}{cluster}(:,lambdaInd) = (test_features_Z_orth * b_grp_HR) * targetHR_sigma + targetHR_mu;% + test_features(:,52);
            %smooth_CV_predicted_BR = movmean(predicted_BR, [4 0]);
            actualHR = test_targetHR;% + test_features(:,52);
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
return;


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
    c_num{cluster,:} = sum(cluster_inds{testDayInd}{testDayInd} == cluster);
    c_p{cluster,:} = c_num{cluster}/length(cluster_inds{testDayInd}{testDayInd});
end

err_final = minLambda_predicted_HR' - actual_testHR;
err_final_rms = rms(err_final);
err_final_nrms = err_final_rms/range(actual_testHR);
err_final_rmsr = rms(err_final./actual_testHR);
err_final_mae = sum(abs(err_final))/length(err_final);
err_final_mape = sum(abs(err_final./actual_testHR))/length(err_final);
err_final_r2 = 1 - sum(err_final.^2)/sum((actual_testHR-mean(actual_testHR)).^2);
err_cp = (actual_testHR - features{testDayInd}(:,52));
err_cp_rms = rms(actual_testHR - features_HR{testDayInd}(:,1));
err_cp_nrmse = err_cp_rms/range(actual_testHR);
err_cp_rmsr = rms(err_cp./actual_testHR);
err_cp_mae = sum(abs(err_cp))/length(actual_testHR);
err_cp_mape = sum(abs(err_cp./actual_testHR))/length(actual_testHR);
err_cp_r2 = 1 - sum(err_cp.^2)/sum((actual_testHR-mean(actual_testHR)).^2);



%% Get list of features used per test day and cluster and optimal features
%used for lambda with min RMSE

clear optimalFeaturesUsed featuresUsed optimalsensorGroupsUsed;
groupInds = 1:max(featureGroupIDs);
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
        %currentReduction{test_section_day}{cluster} = sum(currentSensor .* not(optimalsensorGroupsUsed{test_section_day}{cluster}));
    end
end


%% Plot feature/sensor disappearance
figure(1)
clf

for cluster = 1:clusterNum
    h = subplot(1, clusterNum, cluster);
    feature_exist = featuresUsed{testDayInd}{cluster}';
    feature_exist = [ones(size(feature_exist, 1), 1) feature_exist];
    feature_exist(:, end) = 0;
    feature_lost_table = diff(feature_exist, 1, 2);
    [feature_lost, lambda_lost] = find(feature_lost_table == -1);
    lambda_lost = lambda_lost;
    [~, latest, ~] = unique(feature_lost(end:-1:1), 'stable');
    feature_lost_latest = feature_lost(length(feature_lost) - latest + 1);
    feature_lost_latest = feature_lost_latest(end:-1:1);
    lambda_lost_latest = lambda_lost(length(lambda_lost) - latest + 1);
    lambda_lost_latest = lambda_lost_latest(end:-1:1);
    
    d = [1 ; diff(lambda_lost_latest)] ~= 0;
%    d = [1 ; diff(featureGroupIDs)'] ~=0;
%     d = dl | (~dl & dg);
%     for i = 3:length(d)
%         if (d(i) == 0 && d(i-1) == 0) && (featureGroupIDs(i) ~= featureGroupIDs(i-1))
%             d(i-2) = 1;
%         end
%     end
    [first_feature_lost_ind, ~] = find(d ~= 0);
    first_feature_lost = feature_lost_latest(first_feature_lost_ind);
    first_lambdaInd_lost = lambda_lost_latest(first_feature_lost_ind);
    sensor_lost = groupNames(featureGroupIDs(first_feature_lost));
    
    hold on;

    minLambdaInd = optimal_lambda{testDayInd}{cluster};
    bar(1:sum(first_lambdaInd_lost <= minLambdaInd), lambdas(first_lambdaInd_lost(first_lambdaInd_lost <= minLambdaInd)), 'FaceColor', [1 0.5 0.3]);
    bar((numGroups+1)-sum(first_lambdaInd_lost > minLambdaInd):numGroups, lambdas(first_lambdaInd_lost(first_lambdaInd_lost > minLambdaInd)), 'FaceColor', [0.3 1 0.5]);
    %bar(lambdas(first_lambdaInd_lost), 'FaceColor', [0.3 0.5 1]);
    plot([0 numGroups+1], lambdas(minLambdaInd) * ones(2, 1), 'k--');

    set(h, 'yscale', 'log')
    set(h, 'FontSize', 15);
    set(h, 'XLim', [0 numGroups+1]);
    set(h, 'YLim', [lambdas(1) / 1.1 lambdas(end)]);
    grid on;
    ylabel('\lambda');
    %title(['Group lasso lambda at which sensors are excluded - Cluster = ' num2str(cluster)]);
    title(['Cluster = ' num2str(cluster)]);
    xticks(1:length(first_feature_lost));
    xtickangle(45);
    xticklabels(sensor_lost);

end

%suptitle('Group lasso sensors lambda at which sensors disappear');

%feature_lost_table


%% Plot RMSE and numSensors
figure;
clf;

for test_section_day = 3
    for cluster = 1:cluster_count
%         lambdaInd = optimal_lambda{test_section_day}{cluster};
        %subplot(length(test_section_days), cluster_count, (test_section_day-1)*cluster_count + cluster);
        subplot(cluster_count, 1, cluster);
        yyaxis left
        ylim([-inf max(max(sqrt(avg_CV_MSE_HR{test_section_day}{cluster})),max(sqrt(MSE_HR{test_section_day}{cluster})))]);
        hold on;
        plot(lambdas, sqrt(avg_CV_MSE_HR{test_section_day}{cluster}), 'b');
        plot(lambdas, sqrt(MSE_HR{test_section_day}{cluster}), 'b--');
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
        plot(lambdas, sqrt(avg_CV_MSE_HR{test_section_day}{cluster}), 'b');
        plot(lambdas, sqrt(MSE_HR{test_section_day}{cluster}), 'b--');
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

legend('CV RMSE', 'Test RMSE', 'Num Sensors');

%% Plot NRMSE and numSensors
figure(2);
clf;

for test_section_day = section_days
    for cluster = 1:cluster_count
        subplot(length(test_section_days), cluster_count, (test_section_day-1)*cluster_count + cluster);
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
%NOTE: I messed with the numbers to make it plot the clustering for test
%day = 4
figure
clf;
hold on;
for cluster = 1:clusterNum
    plotData = train_sections_features_Z_reduced(train_cluster_inds{testDayInd} == cluster, :);
    h = plot3(plotData(:,1), plotData(:,2), plotData(:,3), '.', 'MarkerSize', 6);
    %color = get(h, 'Color');
    %plotData = testAccelDataReduced(testClusterInd == cluster, :);
    %plot3(plotData(:,1), plotData(:,2), plotData(:,3), '+', 'MarkerSize', 6, 'Color', color);
end
%plot3(testAccelDataReduced(:,1), testAccelDataReduced(:,2), testAccelDataReduced(:,3), '.', 'MarkerSize', 4);
%legend(cellfun(@num2str, num2cell(reshape(repmat(1:clusterCount, 2, 1)), clusterCount*2, 1), 'uniform', 0));
%legend({'1', '1', '2', '2', '3', '3', '4', '4'})
legend(['1 (' num2str(sum(train_cluster_inds{testDayInd} == 1)) ')'],...
['2 (' num2str(sum(train_cluster_inds{testDayInd} == 2)) ')'],...
['3 (' num2str(sum(train_cluster_inds{testDayInd} == 3)) ')'],...
['4 (' num2str(sum(train_cluster_inds{testDayInd} == 4)) ')']);
title('Clustering Visualization');
axis equal;
set(gca, 'FontSize', 15);
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
figure
clf;
hold on;
%plot(time{dayInd} / 60, IDX{dayInd}, 'o--')
%plot(time{dayInd}(IDX{dayInd} ~= 0) / 60, IDX{dayInd}(IDX{dayInd} ~= 0), '.')
%plot((tags - 1517435703) / 60, 1.5*(ones(13,1)), 'or')

plot(time{testDayInd} / 60, cluster_inds{testDayInd}{testDayInd}, '.')

% moded
% halfWindowSize = 5;
% moded_testClusterInd = [];
% for ind = 1:length(testClusterInd)
%     moded_testClusterInd(ind) = round(mode(testClusterInd(max(1, ind - halfWindowSize):min(end, ind+halfWindowSize))));
% end

%h = plot(time{testDayInd} / 60, moded_testClusterInd, '.')
ax = gca;
ax.YGrid = 'on';
set(gca, 'YLim', [0 clusterNum]);
yticks(0:5);
xlabel('Time (min)');
ylabel('Cluster');
set(gca, 'FontSize', 15);
title('Clusters over time');
%set(gca, 'YTicks', [0 1 2 3 4 5])

%% SVM run training and testing 
% for SVM HR prediction testing for comparison to Group lasso
% Set the svm run flag:
% 1 - Predict Change in HR using lasso optimal features
% 2 - Predict Change in HR using all features
% 3 - Predict Actual HR using lasso optimal features
% 4 - Predict Actual HR using all features

svm_run_flag = 2;
switch svm_run_flag
    case 1 
        svm;
    case 2
        svm_allFeats
    case 3
        svm_hr
    case 4
        svm_allFeats_hr
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



% 
% %% Output train/target data to csv for importing it in R 
% % to use for testing of random forest algorithm comparison vs Group Lasso
% for test_section_day = section_days
%     train_section_days = section_days(section_days ~= test_section_day);
%     for cluster = 1:cluster_count
%         train_features_Z = vertcat(features_Z{train_section_days});
%         train_features_Z = train_features_Z(vertcat(cluster_inds{test_section_day}{train_section_days}) == cluster, :);
%         train_targetBR = vertcat(target_HR{train_section_days});
%         train_targetBR = train_targetBR(vertcat(cluster_inds{test_section_day}{train_section_days}) == cluster, :);
%         test_features = features{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
%         test_targetBR = target_HR{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
%         test_features_Z = features_Z{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
%         
%         train_features_Z_optimal = train_features_Z(:, optimalFeaturesUsed{test_section_day}{cluster});
%         test_features_Z_optimal = test_features_Z(:, optimalFeaturesUsed{test_section_day}{cluster});
%         
%         csvwrite(['train_test_data/train_features_Z_optimal_d' num2str(test_section_day) '_c' num2str(cluster) '.csv'], train_features_Z_optimal);
%         csvwrite(['train_test_data/test_features_Z_optimal_d' num2str(test_section_day) '_c' num2str(cluster) '.csv'], test_features_Z_optimal);
%         csvwrite(['train_test_data/train_targetBR_d' num2str(test_section_day) '_c' num2str(cluster) '.csv'], train_targetBR);
%         
%         %train_features_Z_target_optimal = [train_features_Z_optimal train_targetBR];
%         
%     end
% end
% 
% 
% %% Find the lambda indices given by the increase allowed RMSE amount vector
% 
% for test_section_day = section_days
%     for cluster = 1:cluster_count
%         for increaseAmountInd = 1:length(increaseAmounts)
%             increaseAmount = increaseAmounts(increaseAmountInd);
%             %Find index of lambda with RMSE closest to increased RMSE 
%             [~, increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd)] = min(...
%                 abs(sqrt(avg_CV_MSE_HR{test_section_day}{cluster}(optimal_lambda{test_section_day}{cluster}+1:end)) - ...
%                 (1+increaseAmount)*sqrt(avg_CV_MSE_HR{test_section_day}{cluster}(optimal_lambda{test_section_day}{cluster})))...
%                 );
%             increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd) = ...
%                 increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd) + optimal_lambda{test_section_day}{cluster};
%             %         [~, increaseHRAmountLambdaInd{cluster}(increaseAmountInd)] = min(abs(CV_avg_RMSE_HR{cluster}(minHRCVLambdaInd+1:end) - (1+increaseAmount)*min(CV_avg_RMSE_HR{cluster}(minHRCVLambdaInd+1:end))));
%             %         increaseHRAmountLambdaInd{cluster}(increaseAmountInd) = increaseHRAmountLambdaInd{cluster}(increaseAmountInd) + minHRCVLambdaInd;
%         end
%      end
% end
% 
% %% Calculate average RMSE, number sensors, and cluster counts using the new
% %  increased allowable RMSE percentage lambda index (lambdaInc)
% RMSE_increase = 0.03;
% disp('RMSE for days');
% RMSE_results = [];
% avg_MSE_HR_it = [];
% for test_section_day = section_days
%     %disp(['Day ' num2str(test_section_day)]);
%     %MSE_HR{test_section_day}{cluster}(lambdaInd)
% 
%     %avg_MSE_HR_it(test_section_day) = 0;
%     avg_MSE_HR_it = 0;
%     avg_numSensors_HR = 1;
%     for cluster = 1:cluster_count
%         %disp(['Cluster ' num2str(cluster)]);
% %         lambdaInd = optimal_lambda{test_section_day}{cluster};
%         increaseAmountInd = find(increaseAmounts == RMSE_increase);
%         lambdaInd = increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd);
%         RMSE_results(test_section_day * 3 - 2, cluster) = sqrt(MSE_HR{test_section_day}{cluster}(lambdaInd));
%         RMSE_results(test_section_day * 3 - 1, cluster) = numSensors_HR{test_section_day}{cluster}(lambdaInd);
%         RMSE_results(test_section_day * 3    , cluster) = sum(cluster_inds{test_section_day}{test_section_day} == cluster);
%         
%         %avg_MSE_HR_it(test_section_day) = avg_MSE_HR_it(test_section_day) + MSE_HR{test_section_day}{cluster}(lambdaInd) * sum(cluster_inds{test_section_day}{test_section_day} == cluster);
%         avg_MSE_HR_it = avg_MSE_HR_it + MSE_HR{test_section_day}{cluster}(lambdaInd) * sum(cluster_inds{test_section_day}{test_section_day} == cluster);
%         avg_numSensors_HR = avg_numSensors_HR + numSensors_HR{test_section_day}{cluster}(lambdaInd) * sum(cluster_inds{test_section_day}{test_section_day} == cluster);
%         %disp([num2str(sum(cluster_inds{test_section_day} == cluster)) ' ' num2str(MSE_HR{test_section_day}{cluster}(lambdaInd))]);
%     end
%     %avg_MSE_HR_it(test_section_day) = avg_MSE_HR_it(test_section_day) / length(cluster_inds{test_section_day}{test_section_day});
%     avg_MSE_HR_it = avg_MSE_HR_it / length(cluster_inds{test_section_day}{test_section_day});
%     avg_numSensors_HR = avg_numSensors_HR / length(cluster_inds{test_section_day}{test_section_day});
%     avg_clusterNum = length(cluster_inds{test_section_day}{test_section_day}) / cluster_count;
% 
%     RMSE_results(test_section_day * 3 - 2, cluster_count+1) = sqrt(avg_MSE_HR_it);
%     RMSE_results(test_section_day * 3 - 1, cluster_count+1) = avg_numSensors_HR;
%     RMSE_results(test_section_day * 3    , cluster_count+1) = avg_clusterNum;
% 
% end
% avg_RMSE_HR_it = sqrt(avg_MSE_HR_it);
% avg_RMSE_HR_it = round(avg_RMSE_HR_it' * 1000) / 1000;
% RMSE_results = round(RMSE_results * 1000) / 1000;
% %disp(avg_avg_MSE_HR);
% 
% %% Sensors used for each day using new allowable increased RMSE value 
% RMSE_increase = 0.03;
% sensors_used_results = false(max(featureGroupIDs), length(section_days) * cluster_count);
% for test_section_day = section_days
%     for cluster = 1:cluster_count
% %         lambdaInd = optimal_lambda{test_section_day}{cluster};
%         increaseAmountInd = find(increaseAmounts == RMSE_increase);
%         lambdaInd = increaseHRAmountLambdaInd{test_section_day}{cluster}(increaseAmountInd);
%         
%         %sensors_used_results(sensorGroupsUsed{test_section_day}{cluster}(lambdaInd), (test_section_day-1) * 4 + cluster) = true;
%         sensors_used_results(:, (test_section_day-1) * cluster_count + cluster) = sensorGroupsUsed{test_section_day}{cluster}(lambdaInd, :)';
%         %unique(featureGroupIDs(b_grp_HR ~= 0))
%     end
% end
% 
% sensors_used_results;
% 
%% plot TESTING actual, predicted

figure
clf
hold on;

plot(time{testDayInd}, actual_testHR); % actual future

%{
CP_predicted_BR_comb = [];
time_comb = [];
for cluster = 1:clusterCount
    CP_predicted_BR_comb = [CP_predicted_BR_comb ; CP_predicted_BR{cluster}];
    time_comb = [time_comb ; time{testDayInd}(testClusterInd == cluster)];
end
plot(time_comb/60, CP_predicted_BR_comb); % CP predicted
%}

plot(time{testDayInd}, minLambda_predicted_HR); % predicted
%plot(time{testDayInd}, features_HR{testDayInd}(:,1)); 
grid on;
set(gca, 'FontSize', 15)
title('Heart Rate predction');
legend({'Actual', 'Predicted'});

figure(2); plot(time{testDayInd}/60, err_final, '.');


% %% plot NRMSE and number of features vs lambda for each cluster (NOT USED??)
% 
% % fig=figure
% % clf;
% % %subplotRows = ceil(sqrt(clusterCount));
% % %subplotCols = ceil(clusterCount / subplotRows);
% % subplotRows = 1;
% % subplotCols = clusterCount;
% % 
% % for cluster = 1:clusterCount
% %     h = subplot(subplotRows, subplotCols, cluster);
% %     hold on;
% %     %yyaxis left
% %     plot(lambdas, test_RMSE_BR{cluster}, 'linewidth',3);
% %     %plot(lambdas, train_RMSE_BR{cluster}, 'linewidth',3);
% %     plot(lambdas, CV_avg_RMSE_BR{cluster}, 'linewidth',3);
% %     %plot([min(lambdas) max(lambdas)], [1 1] .* CP_RMSE_BR{cluster}, 'r-');
% %     
% % %     [~, minTestLambdaInd] = min(test_RMSE_BR{cluster});
% % %     plot(lambdas(minTestLambdaInd), test_RMSE_BR{cluster}(minTestLambdaInd), 'diamond', 'MarkerSize', 10);
% % %     
% % %     [~, minTrainLambdaInd] = min(train_RMSE_BR{cluster});
% % %     plot(lambdas(minTrainLambdaInd), train_RMSE_BR{cluster}(minTrainLambdaInd), 'diamond', 'MarkerSize', 10);
% % %     
% % %     [~, minBRCVLambdaInd] = min(CV_avg_RMSE_BR{cluster});
% % %     plot(lambdas(minBRCVLambdaInd), CV_avg_RMSE_BR{cluster}(minBRCVLambdaInd), 'diamond', 'MarkerSize', 10);
% %     
% %     % find lambda corresponding to increased RMSE
% % %     for increaseAmountInd = 1:length(increaseAmounts)
% % %         lambdaInd = increaseBRAmountLambdaInd{cluster}(increaseAmountInd);
% % %         plot(lambdas(lambdaInd), test_RMSE_BR{cluster}(lambdaInd), '^', 'MarkerSize', 10);
% % %     end
% % 
% % %     test_RMSE_BR{cluster}(increaseBRAmountLambdaInd{cluster});
% % %     lambdas(increaseBRAmountLambdaInd{cluster});
% % %     numSensors_BR{cluster}(increaseBRAmountLambdaInd{cluster});
% % %     [~, min10LambdaInd] = min(abs(CV_avg_RMSE_BR{cluster}(minLambdaInd+1:end) - 1.1*min(CV_avg_RMSE_BR{cluster}(minLambdaInd+1:end))));
% % %     min10LambdaInd = min10LambdaInd + minBRCVLambdaInd;
% % %     plot(lambdas(min10LambdaInd), CV_avg_RMSE_BR{cluster}(min10LambdaInd), 'O', 'MarkerSize', 10);
% % 
% %     set(h, 'XLim', [lambdas(1) / 1.4 lambdas(end) * 1.4])
% %     xlabel('\lambda')
% %     ylabel('RMSE')
% % %     yLim = get(gca, 'YLim');
% % %     yLim(1) = 0;
% % %     set(gca, 'YLim', yLim);
% %     %YLim = get(h, 'YLim');
% %     %YLim(1) = 0;
% %     %set(h, 'YLim', YLim)
% %     
% %     yyaxis right
% %     %plot(lambdas, numFeatures_BR{cluster}, '-.', 'linewidth', 3)
% %     %ylabel('Number of features')
% %     h = plot(lambdas, numSensors_BR{cluster}, '-.k', 'linewidth', 3)
% %     ylabel('Number of sensors')
% %     set (gca, 'fontsize', 15)
% %     set(gca, 'xscale', 'log')
% %     set(gca, 'YColor', [0 0 0])
% %     set(gca, 'YLim', [-0.3 8.3])
% %     grid on
% %     %h.YTick = 1:8;
% %     title(['Cluster = ' num2str(cluster)])
% %     %legend('NRMSE', 'Number of features')
% % end
% % lgd = legend('Test', 'LCV', 'No. of Sensors');
% % lgd.FontSize = 10;
% 
% 
% 
% %% plot GCV, SLLCV, LCV NRMSE (NOT USED)
% 
% % figure
% % clf;
% % 
% % hold on;
% % 
% % % GCV
% % [~, minLambda] = min(avg_avg_GCV_RMSE_BR);
% % h1 = plot(lambdas, avg_avg_GCV_RMSE_BR, 'linewidth',3, 'Color', [1 0.4 0]);
% % plot(lambdas(minLambda), avg_avg_GCV_RMSE_BR(minLambda), 'v', 'linewidth',3, 'Color', [1 0.4 0]);
% % 
% % [~, minLambda] = min(CV_avg_avg_RMSE_BR);
% % h2 = plot(lambdas, CV_avg_avg_RMSE_BR, 'linewidth',3, 'Color', [0.4 0.6 0]);
% % plot(lambdas(minLambda), CV_avg_avg_RMSE_BR(minLambda), 'v', 'linewidth',3, 'Color', [0.4 0.6 0]);
% % 
% % set(gca, 'xscale', 'log')
% % xlabel('\lambda')
% % ylabel('RMSE')
% % set(gca, 'XLim', [lambdas(1) / 1.4 lambdas(end) * 1.4]);
% % % yLim = get(gca, 'YLim');
% % % yLim(1) = 0;
% % % set(gca, 'YLim', yLim);
% % set (gca, 'fontsize', 15)
% % grid on
% % legend([h1 h2], 'GCV', 'SLLCV');
% % 
% % %title('Global Cross Validation')
% % %title(['Single \lambda Cross Validation'])
% 
% %% Get min RMSE numbers (NOT USED)
% 
% % for cluster = 1:CV_cluster_count
% %     disp(['cluster ' num2str(cluster) ' (' num2str(sum(cluster==testClusterInd)) ')']);
% %     % LCV
% %     [~, minLambdaInd] = min(CV_avg_RMSE_BR{cluster});
% %     disp(test_RMSE_BR{cluster}(minLambdaInd));
% %     disp(numSensors_BR{cluster}(minLambdaInd));
% %     % GCV
% %     [~, minLambdaInd] = min(avg_avg_GCV_RMSE_BR);
% %     disp(test_RMSE_BR{cluster}(minLambdaInd));
% %     disp(numSensors_BR{cluster}(minLambdaInd));
% %     % SLLCV
% %     [~, minLambdaInd] = min(CV_avg_avg_RMSE_BR);
% %     disp(test_RMSE_BR{cluster}(minLambdaInd));
% %     disp(numSensors_BR{cluster}(minLambdaInd));
% % end
% 
% %% cluster-averaged min RMSE numbers (NOT USED)
% 
% % %LCV
% % LCV_avg_RMSE = 0;
% % for cluster = 1:CV_cluster_count
% %     [~, minLambdaInd] = min(CV_avg_RMSE_BR{cluster});
% %     LCV_avg_RMSE = LCV_avg_RMSE + test_RMSE_BR{cluster}(minLambdaInd) .^ 2 * sum(testClusterInd == cluster);
% % end
% % LCV_avg_RMSE = sqrt(LCV_avg_RMSE / length(testClusterInd));
% % 
% % %GCV
% % [~, minLambdaInd] = min(avg_avg_GCV_RMSE_BR);
% % GCV_avg_RMSE = 0;
% % for cluster = 1:CV_cluster_count
% %     GCV_avg_RMSE = GCV_avg_RMSE + test_RMSE_BR{cluster}(minLambdaInd) .^ 2 * sum(testClusterInd == cluster);
% % end
% % GCV_avg_RMSE = sqrt(GCV_avg_RMSE / length(testClusterInd));
% % 
% % % SLLCV
% % [~, minLambdaInd] = min(CV_avg_avg_RMSE_BR);
% % SLLCV_avg_RMSE = 0;
% % for cluster = 1:CV_cluster_count
% %     SLLCV_avg_RMSE = SLLCV_avg_RMSE + test_RMSE_BR{cluster}(minLambdaInd) .^ 2 * sum(testClusterInd == cluster);
% % end
% % SLLCV_avg_RMSE = sqrt(SLLCV_avg_RMSE / length(testClusterInd));
% % 
% % disp('    GCV       LCV       SLLCV')
% % disp([GCV_avg_RMSE LCV_avg_RMSE SLLCV_avg_RMSE]);
% 
% %% plot error over time (NOT USED)
% 
% % figure
% % clf
% % hold on;
% % %plot(time{testDayInd}/60, target_BR{testDayInd}); % actual future
% % %plot(time{testDayInd}/60, movmean(target_BR{testDayInd}, [4 0]) + features{testDayInd}(testClusterInd == cluster, 53)); % actual future
% % 
% % err = [];
% % for cluster = 1:clusterCount
% %     %[~, optimalLambdaInd] = min(abs(min(train_NRMSE_BR{cluster}) * 1.0 - train_NRMSE_BR{cluster})); % get lambda that is results 10% worse than optimal
% %     %optimalLambdaInd = 121;
% %     [~, minLambdaInd] = min(CV_avg_RMSE_BR{cluster});
% % 
% %     %    plot(time{testDayInd}(testClusterInd == cluster)/60, predicted_BR{cluster}{optimalLambdaInd}); % predicted
% %     %test_actual_BR{cluster}{lambdaInd};
% %     plot(time{testDayInd}(testClusterInd == cluster)/60, predicted_BR{cluster}{minLambdaInd} - test_actual_BR{cluster}{minLambdaInd}, 'k.'); % error
% %     err = [err ; predicted_BR{cluster}{minLambdaInd} - test_actual_BR{cluster}{minLambdaInd}];
% % end
% % 
% % rms(err)
% % 
% % %title('error (predicted - actual)');
% % %legend('1', '2', '3', '4');
% % %legend({'actual', 'CP', 'predicted', 'smooth'});

% %% Try different dimensionality reduction techniques 
% 
% % close all;
% % clc;
% % 
% % techniqueNames = {'PCA', 'LDA', 'MDS', 'ProbPCA', 'FactorAnalysis', 'GPLVM', ...
% %     'Sammon', 'Isomap', 'LandmarkIsomap', 'LLE', 'Laplacian', 'HessianLLE', ...
% %     'LTSA', 'MVU', 'CCA', 'LandmarkMVU', 'FastMVU', 'DiffusionMaps', ...
% %     'KernelPCA', 'GDA', 'SNE', 'SymSNE', 'tSNE', 'LPP', 'NPE', 'LLTSA', ...
% %     'SPE', 'Autoencoder', 'LLC', 'ManifoldChart', 'CFA', 'NCA', 'MCML', 'LMNN'};
% % 
% % clusterFeatureInds = 5:46; % 5:46 5:25
% % % skip 6:9 12:17 21:23 29:31 33:34
% % for techniqueNamesInd = [1:5 10:11 18:20 24:28 32]
% %     disp([num2str(techniqueNamesInd) '/' num2str(numel(techniqueNames)) ' - ' techniqueNames{techniqueNamesInd}]);
% %     [accelDataReduced{testDayInd}, mapping] = compute_mapping(features_Z{testDayInd}(:, clusterFeatureInds), techniqueNames{techniqueNamesInd}, 3);
% %     epsilon = 0.9;
% %     MinPts = 50;
% %     %DBSCAN clustering function found online
% %     trainClusterInd{dayInd} = DBSCAN(accelDataReduced{dayInd}, epsilon, MinPts);
% %     clusterCount{dayInd} = max(trainClusterInd{dayInd});
% %     figure
% %     hold on;
% %     for cluster = 0:clusterCount{dayInd}
% %         plotData = accelDataReduced{dayInd}(trainClusterInd{dayInd} == cluster, :);
% %         plot3(plotData(:,1), plotData(:,2), plotData(:,3), '.', 'MarkerSize', 6);
% %         %plot3(accelDataReduced{dayInd}(:,1), accelDataReduced{dayInd}(:,2), accelDataReduced{dayInd}(:,3), '.');
% %     end
% %     title(techniqueNames{techniqueNamesInd});
% %     legend(['none' cellfun(@num2str, num2cell(1:clusterCount{dayInd}), 'uniform', 0)]);
% %     saveas(gcf,['reduced_' num2str(techniqueNamesInd) '.png']);
% % end

