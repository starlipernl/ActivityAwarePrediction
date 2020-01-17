%Peform HRV analysis
%%
clear all
load('C:\Users\starl\Documents\ARoS_Lab\Power_Op\PowerOptimization-master\Data\y18m02d14\Pre\Data.mat');
%%
data_ecg = bhObj{4}.fetchData('R-R');
time_raw = bhObj{4}.fetchData('TimeSync');
time_raw = time_raw - time_raw(1);
data_ecg = data_ecg./1000;
%data_ecg_mv = (data_ecg - 2048)*0.00625;
%%
settings = InitializeHRVparams('powerop');
[HRV, resultsfile] = Main_HRV_Analysis(data_ecg,time_raw,'RRIntervals', settings);