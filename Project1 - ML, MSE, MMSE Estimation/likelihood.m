function output = likelihood( x_n, sigma_n, t1, t2, pdf1, pdf2 )
range1 = x_n(1:t1-1);
range2 = x_n(t1:t2);
range3 = x_n(t2+1:length(x_n));

log_range1 = log(pdf1(range1, sigma_n));
log_range2 = log(pdf2(range2, sigma_n));
log_range3 = log(pdf1(range3, sigma_n));

output = sum([log_range1, log_range2, log_range3]);

end