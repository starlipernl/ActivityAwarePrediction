% Code to test out various clustering and data reductions including TSNE,
% LLE, KNN, DBSCAN, and analyzing optimal cluster numbers

% tsne reduce dimensionality to 2 and find level sets using KDE
load('phase2_feats_02122019.mat');
load('features_reduced_all_new');
features_red_AE = load('phase2_featsred_02122019', 'features_red');
features_red_AE = features_red_AE.features_red;
%%
% tsne reduction
perp_old = [5 10 20 30 40 50 80 100 150 200];
perp = [60 70 75 85 90 110 120 130 165 180 225 250 275 300 350];
features_reduced2d_new = cell(1,10);
kl_loss_new = cell(1,10);

parfor i = 11:15
    tic
    p = perp(i);
    [features_reduced2d_new{i}, kl_loss_new{i}] = tsne(train_features_Z(1:50000, clusterFeatureInds), 'Perplexity', p, 'Verbose', 1);
    toc
end
save('tsne_results_02122019.mat');

load('phase2_results_02122019.mat', 'train_cluster_inds');
figure; 
for i = 1:10
    for j = 1:6  
        subplot(3,4,i); scatter(features_reduced2d{i}(train_cluster_inds(1:50000)==j,1), ...
        features_reduced2d{i}(train_cluster_inds(1:50000)==j,2), 2, '.')
        hold on
        title(['Perplexity =' num2str(perp_old(i))])
        gcaExpandable;
    end
end
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6')

figure;
for i = 1:10
    subplot(3,4,i); scatter(features_reduced2d{i}(:,1), ...
    features_reduced2d{i}(:,2), 2, '.')
    hold on
    title(['Perplexity =' num2str(perp(i))])
end

%% lle reduction

lle_sub_inds = randperm(size(train_features_Z,1));
[features_reduced_LLE, mapping_LLE] = compute_mapping(...
    train_features_Z(lle_sub_inds(1:10000), clusterFeatureInds), 'LLE', 3, 40);
save('featuresLLE.mat', 'features_reduced_LLE');
scatter3(features_reduced_LLE(:,1), features_reduced_LLE(:,2), features_reduced_LLE(:,3), '.')


%%
% Kernel density estimation (online function)
[bw, kde, x_est, y_est] = kde2d(features_reduced2d{5});
figure;
surf(x_est, y_est, kde);

% plot data densities (matlab function)
figure;
ksdensity(train_feats_red(:,1:2));


% plot tsne-reduced dim data points
figure;
scatter(features_reduced2d(:,1), features_reduced2d(:,2), '.');

%%
% find optimal number of clusters using elbow method and CalinksiHarabasz
clust = zeros(size(train_feats_red,1),20);
sumd = zeros(20,20);
for i=2:20
[clust(:,i), kCents{i}, sumd(1:i,i)] = kmeans(train_feats_red,i,'emptyaction','singleton',...
        'replicate',5);
end
% sum of square errors
k_sse = sum(sumd, 1);
plot(2:20, k_sse(2:20)); title('K-Means Elbow Plot');
xlabel('k (num clusters)'); ylabel('Sum Squared Errors (Dist. from Centroids');
% optimal k by matlab function
va = evalclusters(train_feats_red,clust(:,2:20),'CalinskiHarabasz');

% plot optimal k and reduced dimensionality 
figure;
% optimalK = va.OptimalK;
optimalK = 8;
plt_colors = distinguishable_colors(optimalK);
for j = 1:optimalK
    scatter3(train_feats_red(clust(:,optimalK)==j,1), ...
        train_feats_red(clust(:,optimalK)==j,2), ... 
        train_feats_red(clust(:,optimalK)==j,3), 2, plt_colors(j,:), '.')
    hold on
    title('Optimal Clustering');
end

% Plot optimal clustering from above dim reduction data set on tsne data
% figure;
% for j = 1:va.OptimalK 
%     scatter3(features_reduced2d{5}(lle_sub_inds(clust(:,va.OptimalK)==j),1), ...
%         features_reduced2d{5}(lle_sub_inds(clust(:,va.OptimalK)==j),2), 2, '.')
%     hold on
%     title('Optimal Clustering Tsne');
% end

%% Nearest neighbors search
[knn_idx, knn_D] = knnsearch(features_reduced_LLE, features_reduced_LLE, 'K', 150);
% Plot distances to 40th nearest neighbor in descending order to find elbow
knn_distances = sort(knn_D(:,end), 'descend');
plot(knn_distances)

%% DBSCAN
features_dbscan = features_reduced_LLE;
MinPts = 20;
% Use nearest neighbor distance graph to determine epsilon
% [nn_dist, nnidx] = find_nn(features_dbscan, MinPts);
% eps calculated from elbow plot in Nearest Neighbors search
epsilon = 0.0001;

%DBSCAN clustering function found online
cluster_dbscan = DBSCAN(features_dbscan, epsilon, MinPts);
cluster_dbscan = cluster_dbscan + 1;
cluster_dbscan_cnt = max(cluster_dbscan);

figure;
plt_colors = distinguishable_colors(cluster_dbscan_cnt);
for j = 1:cluster_dbscan_cnt  
    scatter3(features_dbscan(cluster_dbscan==j,1), ...
    features_dbscan(cluster_dbscan==j,2), features_dbscan(cluster_dbscan==j,3), 30, plt_colors(j,:), '.')
    hold on
    title('DBSCAN Clustering');
end

%%
% calculate max lambda
X = orth(train_features_Z);
XY = X' * train_targetHR_Z;
maxlam = max(abs(XY))/length(train_targetHR_Z);
