%Code to analyze at the time series properties of the HR data

%Armaxfilter command uses nonlinear least squares solver in Matlab
%(lsqnonlin). This command uses the Trust-Region-Reflective Least Square
%algorithm. 

%% Initialize variables (must load data first)
clc; 
y = features{4}(:,52);
%y = minLambda_predicted_HR';
y = target_HR{testDayInd};
y_predict = minLambda_predicted_HR;
y_cluster = [];
for i = 1:clusterNum
    y_cluster{i} = y(cluster_inds{testDayInd}{testDayInd}==i);
end
% dy = diff(y); dy = [dy(1) ; dy];
% d2y = diff(dy); d2y = [d2y(1) ; d2y];
% ybar = mean(y);
%y_norm = y-ybar;

%% Plot data, clusters, clustered data
figure(1);
subplot(2,1,1); plot(y); title('Full HR Data');
subplot(2,1,2); plot(cluster_inds{testDayInd}{testDayInd}); title('Cluster Number over Time');
for i = 1:clusterNum
    figure(2); 
    subplot(2,ceil(clusterNum/2),i); plot(y_cluster{i}); title(['Cluster'  num2str(i)]);
end
% figure(3);
% subplot(2,1,1); plot(dy); title('First Difference HR');
% subplot(2,1,2); plot(d2y); title('Second Difference HR');

%-----------------------------------------------------------------------------------
%% Using MFE toolbox from online
%-----------------------------------------------------------------------------------

%% Sample Autocorrelation and partial AC functions
[dy_ac, dy_acstd] = sacf(dy,50);
[dy_pac, dy_pacstd] = spacf(dy,50);
[ac, acstd] = sacf(y,1000);
[pac, pacstd] = spacf(y,50);
[sq_ac, sq_acstd] = sacf(y.^2,1000);

%% Arma filter
C = 1;
P = 1:3;
Q = 0;
forlen = 1;
[parms, ~, ar_err] = armaxfilter(y,C,P,Q);
[aic, bic] = aicsbic(ar_err, C, P,Q);
%[yhattph, yhat, forerr, ystd] = arma_forecaster(y,parms,C,P,Q,10,forlen);
x = 1:length(yhat);

%% Plot ARMA Results
figure(1);
plot(abs(forerr)); title('ARMA Model Forecast Error');
figure(2);
plot(x,yhat,x,yhattph); title('Actual and Forecast HR');
legend('Actual','Forecast');
figure(3);
plot(abs(ar_err)); title('ARMA Model Train Error');
figure(4);
err_arma_actual = abs(actual_testHR-yhattph);
plot(err_arma_actual); title('ARMA to Actual HR Error');

%-----------------------------------------------------------------------------------------------------------------------
%% ARMAX using System ID Toolbox
%------------------------------------------------------------------------------------------------------------------------
%% Estimate ARMAX Model
ex = cell2mat(cluster_inds{testDayInd}(:,testDayInd));
yid = iddata(y,ex);
fl = 30;
na = 3; nb = 2; nc = 0; nk = 0;
% leng = 2000-fl-na;
ymax = armax(yid,[na nb nc nk]);
%% Forecast
% tic
% for i = 1:leng-fl
%      yf = forecast(ymax,yid(i:i+2),fl,yid.u(i+3:i+2+fl));
%      y_predict(i,:) = yf.OutputData(30);
% end
% toc
% start = 2004;
% yf = forecast(ymax,yid(start:start+100),fl,yid.u(start+101:fl));
yf = predict(ymax,yid,fl);

%% Plot
% plot(yid(start:start+131), 'b', yf, 'r');
figure(1); clf;
% yyaxis left;
t = 1:length(yid.OutputData);
plot(t, yid.OutputData, 'b', t, yf.OutputData, 'r');
% y_predict = [NaN(30,1); y_predict(1:end-30)];
% plot(1:length(y_predict),y(1:length(y_predict)),'r', 1:length(y_predict),y_predict,'b');
% err_armax = abs(y(1:length(y_predict))-y_predict);
% err_armax_rms = rms(err_armax);
err_armax = abs(yid.OutputData-yf.OutputData);
err_armax_rms = rms(err_armax);
% figure(2);
% % yyaxis right;
% plot(err_armax);
%Use for modeling the delta target hr
HR_real = target_HR{testDayInd} + features{testDayInd}(:,52);
HR_out = yid.OutputData + features{testDayInd}(:,52);
HR_predict = yf.OutputData + features{testDayInd}(:,52);
HR_err = abs(HR_real - HR_predict);
plot(HR_err);
err_HR_rms = rms(HR_err);
figure(2)
plot(t, HR_real, 'b', t, HR_predict, 'r');
%--------------------------------------------------------------------------------------------------------------------
%% Arimax Using Econometrics Toolbox
%--------------------------------------------------------------------------------------------------------------------
ex = cell2mat(cluster_inds{4}(:,4));

P = 4; D = 0; Q = 2;
yest = y(P+D+1:end,:);
y0 = y(1:P+D,:);
ymodelx = arima(P,D,Q);
%ymodelx.Variance = garch(2,2);
%ymodelx.Distribution = 't';
yfit = estimate(ymodelx,yest,'X',ex, 'Y0', y0);
[E,V] = infer(yfit, y);
[yhat, Ehat, Vhat] = filter(yfit, y_predict);
figure(1); plot(E);
% figure(2); plot(V);
% se = sacf(stdE,20);
% spe = spacf(stdE,20);
% figure;
% histogram(stdE,10)


%% forecast
len = 2000;
fl = 10;
% y_predict = zeros(len,1);
tic
for i = 1:len
    [Yfore, YMSE] = forecast(yfit,fl,'Y0',y(1:P+D+i-1), 'X0', ex(1:P+D+i-1,:), 'XF', ex(P+D+i:P+D+i+fl));
    y_predict(i,:) = Yfore(fl);
end
toc
%% Plot
xax = 1:len;
xax=xax';
y_compare = y(P+D+fl+1:P+D+fl+len);
err_armaxgarch = abs(y_compare - y_predict);
plot(xax, y_compare, 'b',xax,y_predict, 'r');
% for i = 1:leng-fl
%      yf = forecast(yfit,fl(i:i+2),fl,yid.u(i+3:i+2+fl));
%      y_predict(i,:) = yf.OutputData(30);
% end













