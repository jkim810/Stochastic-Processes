%% Part 1.
clear all, clc;
%% part 1.a.
itr = 1000;
var_x = 3;
sig_x = sqrt(var_x);
SNR = 1;
A = SNR*var_x;

not_present = normrnd(0, sig_x, 0.8*itr, 1);
present = normrnd(A, sig_x, 0.2*itr,1);

%calculate for when signal present
%correct
P_present = normpdf(present,A,sig_x);
%incorrect
P_absent = normpdf(present,0,sig_x);
error = P_present*0.2 - P_absent*0.8; 

%calculate for when signal not present
%incorrect
P_present = normpdf(not_present,A,sig_x);
%correct
P_absent = normpdf(not_present,0,sig_x);
error2 = P_absent*0.8 - P_present*0.2;

s=sign([error;error2]);
P_error = sum(s(:)==-1)/itr

MAP = A/2 + var_x*log(4)/A;
P_Present = normcdf(MAP,A,sqrt(var_x));
P_Absent = 1 - normcdf(MAP,0,sqrt(var_x));
P_error_theoretical = P_Present*0.2 + P_Absent*0.8

%% part 1.b.
itr = 1000;
var_x = 3;
SNR1 = 2;
SNR2 = 1;
SNR3 = 0.5;
[true_positive1,false_positive1] = ROC(SNR1, var_x, itr);
[true_positive2,false_positive2] = ROC(SNR2, var_x, itr);
[true_positive3,false_positive3] = ROC(SNR3, var_x, itr);
plot(smooth(false_positive1),smooth(true_positive1), 'r', smooth(false_positive2),smooth(true_positive2), 'g',smooth(false_positive3),smooth(true_positive3), 'b')
xlabel('False Positive')
ylabel('True Positive')
title('Simlation of ROCs for Various SNRs')
legend('SNR = 2', 'SNR = 1', 'SNR = 0.5')

%% part 1.c.


%% Part 3
iris = load('iris.mat');

idx = randperm(150);
iris_features = iris.features(idx,:);
iris_labels = iris.labels(idx);

train_features = iris_features(1:75,:);
train_labels = iris_labels(1:75);
test_features = iris_features(76:end,:);
test_labels = iris_labels(76:end);

