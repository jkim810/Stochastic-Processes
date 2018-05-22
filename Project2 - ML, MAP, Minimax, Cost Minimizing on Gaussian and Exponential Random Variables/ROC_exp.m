function [ result, correct_result, true_positive, false_positive] = ROC_exp( rate1, rate2 )
 
r = randi(10000,2,1);
 
p1 = round(exprnd(rate1,r(1),1));
p2 = round(exprnd(rate2,r(2),1));
symbols = [zeros(r(1),1)+1;zeros(r(2),1)+2];
tmp = [p1;p2];
idx = randperm(sum(r));
rcvd = tmp(idx);
correct_result = symbols(idx);
 
MAP1 = exppdf(rcvd,rate1);
MAP2 = exppdf(rcvd,rate2);
MAP = [MAP1 MAP2];
[row, col] = find(MAP==max(MAP,[],2));
result = sortrows([row, col]);
result = result(:,2);
 
threshold = linspace(min(tmp),max(tmp), max(tmp)-min(tmp)+1);
curve = repmat(threshold, sum(r),1) - repmat(rcvd, 1, max(tmp)-min(tmp)+1);
prediction = sign(curve)==1;
correct = repmat(correct_result,1,length(threshold))==1;
true_positive_curve = prediction & correct;
false_positive_curve = prediction & not(correct);
true_positive = sum(double(true_positive_curve));
false_positive = sum(double(false_positive_curve));
true_positive = true_positive/max(true_positive);
false_positive = false_positive/max(false_positive);
 
end
