function [train_cluster_inds, test_cluster_inds] = runCluster(trainFeats_red, testFeats_red, k)
% function to cluster reduced dimensionality features into k clusters using
% k-means, returns: train and test cluster indices
% cluster reduced training data
[train_cluster_inds, cluster_centroids] = kmeans(trainFeats_red, k);
% sort clusters so that lower cluster number has more points
clusterSize = [];
for cluster = 1:k
    clusterSize(cluster) = sum(train_cluster_inds == cluster);
end
[~, ind] = sort(clusterSize, 'descend');
new_cluster_inds = [];
new_centroids = [];
for cluster = 1:k
    new_cluster_inds(train_cluster_inds == ind(cluster)) = cluster;
    new_centroids(cluster, :) = cluster_centroids(ind(cluster), :);
end
train_cluster_inds = new_cluster_inds;
cluster_centroids = new_centroids;
% cluster_inds : clustering based on test_section_day (1st index) cluster 
% centroids and applied to each day (2nd index)
test_cluster_inds = kmeans(testFeats_red, k, 'Start', cluster_centroids);