clc;
DATA_PROTOCOL = 'C:\Users\starl\Google Drive\VolunteerDataFall2018\';
ai_allSessions = [];

sessions = [11:15];
data = [];
for i = sessions
    folderInput = [DATA_PROTOCOL '8702\Session' num2str(i) '\Post'];
    folderOutput = [DATA_PROTOCOL '8702\Session' num2str(i) '\Post'];
    disp('Starting')
    disp(folderInput);
    try
    load([folderInput '\' 'DataFeatures.mat']);
    data = [data; x(:, [6 9 11])];
    catch e
        disp('Error in ')
        disp(folderInput);

        fprintf(1,'The identifier was:\n%s',e.identifier);
        fprintf(1,'There was an error! The message was:\n%s',e.message);
        fprintf('\n');
        
    end
    disp('---------------------------------------------------------')
end

ai = getActivityIndex(data);
ai_allSessions = [ai_allSessions;ai];

histogram(ai_allSessions, 200)
title('Activity Index Histogram Subject 4305 (20 hours)')