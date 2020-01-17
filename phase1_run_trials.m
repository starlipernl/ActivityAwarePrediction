% iterate through deltah values and run scripts for both clustering
% and no clustering then save workspace to file
tic
currentfolder = pwd;
if isfile([currentfolder '\results\delHR_10s.mat'])
    print('YOU ARE TRYING TO OVERWRITE FILES NO NO NO NO');
    return;
end

for script_num = 1:4
    % run with 4 clusters
    clearvars -except script_num;
    delh = [10 30 60 90];
    currentfolder = pwd;
    clusterNum = 4;
    FL = delh(script_num);
    phase1_deltaHR_main
    filename = [currentfolder '\results\delHR_' num2str(FL) 's.mat'];
    save(filename);
    
    % run with no clustering
    clearvars -except script_num;
    delh = [10 30 60 90];  
    currentfolder = pwd;
    clusterNum = 1;
    FL = delh(script_num);
    phase1_deltaHR_main
    filename = [currentfolder '\results\delHR_' num2str(FL) 's_nocluster.mat'];
    save(filename);
end
toc