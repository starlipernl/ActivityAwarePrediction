function [train_feats_red, test_feats_red] = dimReduce(trainFeats, testFeats)

%dimensionality reduction toolbox found online
%reducedDimData = compute_mapping(, 'LPP');
[train_feats_red, mapping] = compute_mapping(trainFeats, 'LPP', 3, 25);

%out_of_sample function of dim reduction toolkit for applying
%previously calculated dim reduction mapping using training
%features on test day
test_feats_red = out_of_sample(testFeats, mapping);

