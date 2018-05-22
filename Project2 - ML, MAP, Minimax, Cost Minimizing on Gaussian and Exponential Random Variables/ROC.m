function [ true_positive, false_positive ] = ROC( SNR, var_x, itr )

A = SNR*var_x;
sig_x = sqrt(var_x);

not_present = normrnd(0, sig_x, 0.8*itr, 1);
present = normrnd(A, sig_x, 0.2*itr,1);

%calculate for when signal present
%correct
P_present = normpdf(present,A,sig_x);
%incorrect
P_absent = normpdf(present,0,sig_x);
true_positive = P_present*0.2 - P_absent*0.8; 

%calculate for when signal not present
%incorrect
P_present = normpdf(not_present,A,sig_x);
%correct
P_absent = normpdf(not_present,0,sig_x);
false_positive = P_present*0.2 - P_absent*0.8;

lowbound_threshold = min([true_positive; false_positive]);
highbound_threshold = max([true_positive; false_positive]);
threshold = linspace(lowbound_threshold,highbound_threshold,100);
true_positive_curve = repmat(true_positive,1,100) - repmat(threshold,200,1);
false_positive_curve = repmat(false_positive,1,100) - repmat(threshold,800,1);
true_positive_curve = sign(true_positive_curve)==1;
false_positive_curve = sign(false_positive_curve)==1;
true_positive = sum(true_positive_curve(1:end,:))/200;
false_positive = sum(false_positive_curve(1:end,:))/800;

end

