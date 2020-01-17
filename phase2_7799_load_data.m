% Script to load data, split into training/testing sets and assign
% cross validation fold indices

clear; clc;
tic
% Parameters
clusterFeatureInds = 3:23; % wrist IMU features
% PARSE DATA
working_dir = pwd;
dataDir = [working_dir '\VolunteerDataFall2018\8702'];
% find all directories in data folder
allFiles = dir(dataDir);
allDirs = allFiles([allFiles(:).isdir]);
% remove hidden files
sessionDirs = {allDirs(~ismember({allDirs.name}, {'.', '..'})).name};
numDirs = numel(sessionDirs);
% initialize variables for data import
time = {};
features = {};
target_HR = {};
features_HR = {};
delta_HR = {};
% sessionNums = [2:4, 7, 9:10,12, 14:15, 17:20, 22, 26:33, 35];
sessionNums = [5, 6, 8, 11:15];
sessions = [];
% cycle through all session folders and load valid data 
for dayInd = sessionNums
    fname = [dataDir '\' 'Session' num2str(dayInd) '\Post\DataFeaturesAB.mat'];
    if isfile(fname)
        load([dataDir '\' 'Session' num2str(dayInd) '\Post\DataFeaturesAB.mat']) % loads vars into variables  t x y
        if size(y,1) < 100 % do not load empty data 
            continue;
        end
    else
        continue;
    end
    % vector of valid data sessions loaded
    sessions = [sessions dayInd];
    % smooth HR data (remove spikes)
%     y = movmean(y(:,1), [4 0], 'omitnan');
    dayData = x(:,1:44);
    % calculate first and second diff HR features
    dHR = diff(y); dHR = [dHR(1) ; dHR];
    d2HR = diff(dHR); d2HR = [d2HR(1) ; d2HR];
    % shift the data according to the delh value (delta history window)
    featsHR = [y dHR d2HR];
    time{end+1} = t;
    features{end+1} = dayData;
    features_HR{end+1} = featsHR;
    target_HR{end+1} = y;
    % Fix nan in features
    for col=1:size(features{end},2)
        nanx = isnan(features{end}(:,col));
        t_inter = 1:numel(features{end}(:,col));
        features{end}(nanx,col) = interp1(t_inter(~nanx), x(~nanx), t_inter(nanx));
    end
end
% number of sessions of usable data
n_sess = length(sessions);

% Split data 70/30 training/testing based on sessions (this ensures no data
% from a session is in both training and testing set because data within
% sessions are highly correlated
dataSplit = 0.7; % 70% split
train_sess = false(n_sess, 1); % initialize train/test vector
train_sess(1:round(dataSplit*n_sess)) = true; % set 70% true
train_sess = train_sess(randperm(n_sess)); % randomize
% set train and test sets according to randomized split
trainFeats = features(train_sess);
testFeats = features(~train_sess);
trainFeatsHR = features_HR(train_sess);
testFeatsHR = features_HR(~train_sess);
trainTarget = target_HR(train_sess);
testTarget = target_HR(~train_sess);
trainTime = time(train_sess);
testTime = time(~train_sess);

%% CV index assignment
%assign CV kfold indices
numFolds = 4;
idxCV = crossvalind('Kfold', size(trainFeats,2), numFolds);

save('prepared_data_8702');
