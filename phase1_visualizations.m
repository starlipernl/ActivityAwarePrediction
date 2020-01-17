% Script to plot the metrics vs the history window length (delta_h)

% Data matrices ordered as follows:
% - delta_h(1, :) ==> delta_h = [10 30 60 90]
% - x(1:7, :) ==> Cons Pred, Group Lasso C, Group Lasso, SVM-Lasso-C,
%                 SVM-Lasso, SVM-C, SVM for each delta_h respectively
% Where C specifies activity clustering used and SVM-Lasso specifies that
% the reduced feature set found by group lasso used in SVM

clearvars; clc; 
delta_h = [10 30 60 90];
increaseAmounts = {[linspace(0,2,20)]
                   [linspace(0,10,50)]
                   [linspace(0,20,100);]
                   [linspace(0,20,100)]};
currentSensor = [0.312 0.263, 0.263, 0.312, 3.191, 0.312, 2.673];
currentWrist = 4.126;
currentChest = 5.819;
currentTotal = currentWrist + currentChest;
currentFolder = pwd;
datasource = [currentFolder '\results\K_10invsqrtP\'];


%% Load Metrics
sensors_used = zeros(5*length(delta_h), 7);
for ii = 1:length(delta_h)
    del = delta_h(ii);
    %Load data for clustering
    filename = [datasource 'delHR_' num2str(del) 's.mat'];
    load(filename, 'res_cp', 'results',  'optimalsensorGroupsUsed', 'svm_results', 'svm_results_allFeats');
    %Load data without clustering
    filename = [datasource 'delHR_' num2str(del) 's_nocluster.mat'];
    results_nc = load(filename, 'res_cp', 'results',  'optimalsensorGroupsUsed', 'svm_results', 'svm_results_allFeats');
    %Construct metrix matrix for plotting
    sensors_used((ii-1)*5+1:(ii-1)*5+4, :) = [optimalsensorGroupsUsed{3}{1}; optimalsensorGroupsUsed{3}{2}; ...
        optimalsensorGroupsUsed{3}{3}; optimalsensorGroupsUsed{3}{4};];
    sensors_used((ii-1)*5+5,:) = results_nc.optimalsensorGroupsUsed{3}{1};
    mae(:, ii) = [res_cp.mae; results.mae; results_nc.results.mae; svm_results.mae; ... 
        results_nc.svm_results.mae; svm_results_allFeats.mae; results_nc.svm_results_allFeats.mae];
    rmse(:, ii) = [res_cp.rms; results.rms; results_nc.results.rms; svm_results.rmse; ... 
        results_nc.svm_results.rmse; svm_results_allFeats.rmse; results_nc.svm_results_allFeats.rmse];
    nrmse(:, ii) = [res_cp.nrms; results.nrms; results_nc.results.nrms; svm_results.nrmse; ... 
        results_nc.svm_results.nrmse; svm_results_allFeats.nrmse; results_nc.svm_results_allFeats.nrmse];
    r_squared(:, ii) = [res_cp.r2; results.r2; results_nc.results.r2; svm_results.r2; ... 
        results_nc.svm_results.r2; svm_results_allFeats.r2; results_nc.svm_results_allFeats.r2];
end

%% Plot Metrics
%Plot Mean Absolute Error (MAE)
load('metrics_delh');
plots = [1 2 4 5];
figure(1)
subplot(2,2,1);
hold on
for ii = plots %1:size(mae,1)
    plot(delta_h, mae(ii,:), 'LineWidth', 2, 'Marker', 'x', 'MarkerSize', 10)
end
xlabel('\delta_h  (s)');
ylabel('MAE (bpm)');
xticks([10 30 60 90]);
title('(a) Mean Absolute Error vs \delta_h');
% legend('CP', 'GL-C', 'GL', 'SVM-L-C', 'SVM-L', 'SVM-C', 'SVM', 'location', 'northwest');
legend('CP', 'GL-C', 'SVM-L-C', 'SVM-L', 'location', 'northwest');
set(gca, 'fontsize', 12)
% saveas(figure(1), 'mae_delh.jpg')
hold off

% Plot RMSE
figure(1)
subplot(2,2,2);
hold on
for ii = plots %1:size(rmse,1)
    plot(delta_h, rmse(ii,:), 'LineWidth', 2, 'Marker', 'x', 'MarkerSize', 10)
end
xlabel('\delta_h (s)');
ylabel('RMSE');
xticks([10 30 60 90]);
title('(b) Root Mean Square Error vs \delta_h');
%legend('CP', 'GL-C', 'GL', 'SVM-L-C', 'SVM-L', 'SVM-C', 'SVM', 'location', 'northwest');
set(gca, 'fontsize', 12)
% saveas(figure(4), 'rmse_delh.jpg')
hold off

% Plot NRMSE
figure(1)
subplot(2,2,3); 
hold on
for ii = plots %1:size(nrmse,1)
    plot(delta_h, nrmse(ii,:), 'LineWidth', 2, 'Marker', 'x', 'MarkerSize', 10)
end
xlabel('\delta_h (s)');
ylabel('NRMSE');
xticks([10 30 60 90]);
title('(c) Normalized Root Mean Square Error vs \delta_h');
% legend('CP', 'GL-C', 'GL', 'SVM-L-C', 'SVM-L', 'SVM-C', 'SVM', 'location', 'northwest');

set(gca, 'fontsize', 12)
% saveas(figure(2), 'nrmse_delh.jpg')
hold off

% Plot R-Squared
figure(1)
subplot(2,2,4);
hold on
for ii = plots %1:size(r_squared,1)
    plot(delta_h, r_squared(ii,:), 'LineWidth', 2, 'Marker', 'x', 'MarkerSize', 10)
end
xlabel('\delta_h (s)');
ylabel('R^{2}');
xticks([10 30 60 90]);
title('(d) R^{2} vs \delta_h');
%legend('CP', 'GL-C', 'GL', 'SVM-L-C', 'SVM-L', 'SVM-C', 'SVM', 'location', 'southwest');
set(gca, 'fontsize', 12)
hold off

%saveas(figure(1), 'metrics_delh.jpg')


%% Plot accuracy-power trade-off
currentFolder = pwd;
clear currentReductions;
subtitles = ['a' 'b' 'c' 'd'];
fc = figure;
fa = figure;
for ii = 1:length(delta_h)
    del = delta_h(ii);
    %Load current reductions data for clustering
    filename = [datasource 'delHR_' num2str(del) 's.mat'];
    load(filename, 'current_reduction_new');
    currentReductions{ii}(:, 1:5) = current_reduction_new/currentTotal * 100;
    %Load current reductions data without clustering
    filename = [datasource 'delHR_' num2str(del) 's_nocluster.mat'];
    load(filename, 'current_reduction_new');
    currentReductions{ii}(:, 6) = current_reduction_new(:, 1)/currentTotal * 100;
    
    %Plot the data
%     figure;
    figure(fc);
    subplot(2,2,ii); xlabel('Increase in RMSE'); 
    ylabel('Current Reduction (%)');
    title(['(' subtitles(ii) ')']);
    hold on; 
    for j = 1:4   
        plot(increaseAmounts{ii}, currentReductions{ii}(:,j), 'linewidth', 2);
    end
    hold off;
    legend('I_{c_1}', 'I_{c_2}', 'I_{c_3}', 'I_{c_4}', 'Fontsize', 12);
    set(gca, 'Fontsize', 12);
    ylim([0 100]);
    figure(fa);
    subplot(2,2,ii);
    hold on;
    plot(increaseAmounts{ii}, currentReductions{ii}(:,5), 'linewidth', 2);
    plot(increaseAmounts{ii}, currentReductions{ii}(:,6), 'linewidth', 2);
    hold off;
    xlabel('Increase in RMSE');
    ylabel('Current Reduction (%)');
    title(['(' subtitles(ii) ')']);
    ylim([0 100]); %max(currentReductions{ii}(1:length(increaseAmounts),5:6), [], 'all')]);
    set(gca, 'Fontsize', 12);
end
legend('I_{avg}', 'I_{nc}', 'Fontsize', 12);
% legend('I_{c_1}', 'I_{c_2}', 'I_{c_3}', 'I_{c_4}', 'I_{avg}', 'I_{c_0}', 'Fontsize', 14);


