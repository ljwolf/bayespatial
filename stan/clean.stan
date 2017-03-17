
data {
    int<lower = 1> n;
    int<lower = 1> p;
    matrix[n,p] X;
    vector[n] y;
}
parameters{
  real constant;
  vector[p] beta;
  real<lower=0>tau;
}

model{
  constant ~ normal(0,1);
  beta ~ normal(0,1);
  tau ~ cauchy(0, 5);
  y ~ normal(X * beta + constant, tau);
}