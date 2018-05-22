%% ECE302 MATLAB Exercise: Estimation Techniques
% David Kim, Junbum Kim

%% SCENARIO1

clc; clear all; close all;

itr = 1000;
measurements = 200;
mu_v = 0;
var_v = 3;
sig_v = sqrt(var_v);
v = normrnd(mu_v, sig_v, itr, measurements);

%% 1.a
% since both v and theta have distinct mean and variances, assume
% independence. Given that the two Gaussian random variables are
% independent, the two variables are jointly Gaussian. For jointly Gaussian
% random variables, the mmse estimator is linear.
mu_theta = 10;
var_theta = 5;
theta = normrnd(mu_theta, sqrt(var_theta), itr, 1);
theta = repmat(theta,1,measurements);
h = 0.5;
x =  h*theta + v;
mf = repmat(1:measurements,itr,1);
x = cumsum(x,2)./mf;
theta_mmse = mu_theta + repmat(h*var_theta./(h^2*var_theta+var_v./(1:measurements)),itr,1).*(x-h*mu_theta);   % MMSE of theta at corresponding iteration and measurement.
mmse= mean((theta-theta_mmse).^2);   % MSE for MMSE 

%% 1.b

% MLE of theta is value which maximizes the probability of observations of
% X made given the distribution of v. This is obtained by finding the
% maximum of log likelihood function, i.e. zero derivative.
% For this case,it is found MLE @ nth measurement = mean of first n measurements/h.

theta_mle = cumsum(x/h,2)./mf;
mle = mean((theta-theta_mle).^2);

%% 1.c

plot(1:measurements,mmse,1:measurements,mle);
title('MSE of MMSE and MLE');
xlabel('Number of Measurements')
ylabel('MSE')
legend('MMSE','MLE')
ylim([0 5])

% It is observed that MMSE method is much less sensitive to poor SNR
% compared to MLE method.

%% 1.d

% Assume two wrong priors with same mean: 1. Uniform distribution, 2.
% Exponential distribution

theta_u = 2*mu_theta*rand(itr,1);   
theta_u = repmat(theta_u,1,measurements);       % Uniform distribution

theta_exp = exprnd(mu_theta,itr,1);
theta_exp = repmat(theta_exp,1,measurements);    % Exponential distribution

mmse_u= mean((theta_u-theta_mmse).^2);      % MSE for MMSE with Uniform as wrong prior
mmse_exp= mean((theta_exp-theta_mmse).^2);  % MSE for MMSE with Exponential as wrong prior

figure;
plot(1:measurements,mmse,1:measurements,mmse_u,1:measurements,mmse_exp);
title('MSE for MMSE with wrong priors');
xlabel('Number of Measurements')
ylabel('MSE')
legend('MMSE - Correct Prior', 'MMSE - Incorrect Prior (Uniform)', 'MMSE - Incorrect Prior (Exponential)');

% It is worth noting that more measurements do not improve the MSE if the
% give prior is incorrect, which makes sense since the information obtained
% from measurements is incorrect.

%% SCENARIO 2

% pdf1 = @(x_n, sigma_n) 0.5*(normpdf(x_n,-1,sigma_n)+normpdf(x_n,1,sigma_n));
% pdf2 = @(x_n, sigma_n) 0.25*(normpdf(x_n,-2,sigma_n)+normpdf(x_n,2,sigma_n)+2*normpdf(x_n,0,sigma_n));
% For some reason, using the normpdf function drastically increases the
% execution time, so the following equations are used as pdf.

pdf1 = @(x_n, sigma_n) 0.5 * 1/(sqrt(2*pi)*sigma_n) * (exp(-(x_n - 1).^2/(2*sigma_n^2)) + exp(-(x_n + 1).^2/(2*sigma_n^2)));
pdf2 = @(x_n, sigma_n) 0.5 * 1/(sqrt(2*pi)*sigma_n) * exp(-(x_n).^2/(2*sigma_n^2)) + 0.25 * 1/(sqrt(2*pi)*sigma_n)*(exp(-(x_n - 2).^2/(2*sigma_n^2)) + exp(-(x_n + 2).^2/(2*sigma_n^2)));

n = 100;
t1 = 10;
t2 = 80;

nvar = logspace(-1.5, 3.5, 40);
p = zeros(1, length(nvar));
net_cost = zeros(1, length(nvar));

for i = 1:length(nvar)
    sigma_n = sqrt(nvar(i));
    trials = 100;
    count = 0;
    cost = 0;
    
    for j = 1:trials
        signal = randi(2,1,n);
        signal(signal == 2) = -1;
        interference = zeros(1,n);
        interference_add = randi(2,1,t2-t1+1);
        interference_add(interference_add == 2) = -1;
        interference(t1:t2) = interference_add;
        
        noise = randn([1,n]) * sigma_n;
        recv = signal + interference + noise;
        
        ml = -1000000;
        max_t1 = -1;
        max_t2 = -1;
        
        for t1e = 1:n-9         % interference range is 10-80 symbols
            for t2e = t1e+9:n
                if t2e-t1e>79
                    break
                end
                max_likelihood = likelihood(recv, sigma_n, t1e, t2e, pdf1, pdf2);
                if max_likelihood > ml
                    ml = max_likelihood;
                    max_t1 = t1e;
                    max_t2 = t2e;
                end
            end
        end
        
        if(max_t1 == t1) && (max_t2 == t2)
            count = count + 1;
        else
            cost = abs(max_t1 - t1) + abs(max_t2 - t2);
        end
        net_cost(i) = net_cost(i)+cost;
    end
    net_cost(i) = net_cost(i)/trials;
    p(i) = count/trials;
end

SNR = 1./nvar;
figure;
plot(SNR, p);
xlabel('SNR');
ylabel('Probability');
title('Probability of Success vs SNR');
figure;
semilogy(SNR, net_cost);
xlabel('SNR');
ylabel('Cost (logscale)');
title('Linear Cost on Logscale vs SNR');


