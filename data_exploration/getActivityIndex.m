% get activity index for each second
function act_ind = getActivityIndex(x)
%get the variance of the accel when its still
xstd = sqrt(x);

totalvar = mean(x,2);
[~, idx] = sort(totalvar);
idxrest = idx(1:round(length(totalvar)/100));
minStd = mean(xstd(idxrest,:),1); 

%min_var = mean(min(xvar));
% min_var = min(xstd);
%min_var = min_var.^2;

act_ind = zeros(length(xstd),1);

for t = 1:length(xstd)
    sum1 = 0;
    for i= 1:3
        sum1 = (xstd(t,i) - minStd(i))/minStd(i) + sum1;
        
    end
    sum1 = sum1/3;
    max_val = max(sum1,0);
    act_ind(t) = max_val;
end
