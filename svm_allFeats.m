%% SVM training prep 
% for SVM HR prediction testing for comparison to Group lasso

for test_section_day = 3%section_days
    train_section_days = section_days(section_days ~= test_section_day);
    for cluster = 1:cluster_count
        train_features_Z = vertcat(features_Z{train_section_days});
        train_features_Z = train_features_Z(vertcat(cluster_inds{test_section_day}{train_section_days}) == cluster, :);
        train_targetHR = vertcat(target_HR{train_section_days});
        train_targetHR = train_targetHR(vertcat(cluster_inds{test_section_day}{train_section_days}) == cluster, :);
        test_features = features{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
        test_targetHR = target_HR{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
        test_features_Z = features_Z{test_section_day}(cluster_inds{test_section_day}{test_section_day} == cluster, :);
        
        train_features_Z_optimal = train_features_Z;
        test_features_Z_optimal = test_features_Z;
        
        train_features_Z_target_optimal = [train_features_Z_optimal train_targetHR];
        
        %return;

        % SVM Testing
        trainRegressionModel = str2func('trainRegressionModel_auto');
        %trainRegressionModel = str2func(['trainRegressionModel_d' num2str(test_section_day) '_c4']);
        %trainRegressionModel = str2func(['trainRegressionModel_d' num2str(day) '_p' num2str(predictModelType)]);
        [trainedModel, ~] = trainRegressionModel(train_features_Z_target_optimal);

        svm_predicted_HR{test_section_day}{cluster} = trainedModel.predictFcn(test_features_Z_optimal) + test_features(:, 52);
        svm_smooth_predicted_HR{test_section_day}{cluster} = movmean(svm_predicted_HR{test_section_day}{cluster}, [4 0]);
        svm_actual_HR{test_section_day}{cluster} = test_targetHR + test_features(:, 52);
        
        svm_RMSE{test_section_day}{cluster} = rms(...
            svm_actual_HR{test_section_day}{cluster} - ...
            svm_predicted_HR{test_section_day}{cluster}...
            );

%         figure
%         clf
%         hold on;
%         plot(svm_actual_HR{test_section_day}{cluster}, 'b');
%         plot(svm_predicted_HR{test_section_day}{cluster}, 'r');
%         plot(svm_smooth_predicted_HR{test_section_day}{cluster}, 'c');
        
    end
    
end


round([svm_RMSE{3}{:}] * 1000) / 1000;