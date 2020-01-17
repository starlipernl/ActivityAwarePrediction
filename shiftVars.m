function [vars_shift, target_shift, c_shift] =  ... 
    shiftVars(t, vars, varsHR, target, c_inds, delh)

% find any gaps in time greater than 5 seconds (session stops)
sessionBreaks = find(diff(t) > 5);
breakStarts = [1 ; sessionBreaks + 1];
breakEnds = [sessionBreaks ; numel(t)];
clust_inds(1,:) = c_inds;
vars_shift = [];
varsHR_shift = [];
c_shift = [];
target_shift = [];

for breakIdx = 1:numel(breakStarts)
    if breakEnds(breakIdx) < breakStarts(breakIdx) + delh
        continue;
    else
        vars_shift = [vars_shift ; vars(breakStarts(breakIdx)+ ... 
            delh:breakEnds(breakIdx), :)];
        varsHR_shift = [varsHR_shift ; ...
            varsHR(breakStarts(breakIdx):breakEnds(breakIdx)-delh, :)];
        target_shift = [target_shift ; ... 
            target(breakStarts(breakIdx)+delh:breakEnds(breakIdx),:)];
        c_shift = [c_shift clust_inds(breakStarts(breakIdx)+ ... 
            delh:breakEnds(breakIdx))];        
    end
end

vars_shift = [vars_shift varsHR_shift];