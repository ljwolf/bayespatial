data {
  int<lower=0> n_obs; // number of observations, here 56
  int<lower=0> counts[n_obs]; // vector of outcomes, which are counts of lip cancer in this case
  vector[n_obs] exposure; // X variable in the model, modelling exposure to UV rays
  matrix[n_obs, n_obs] W; // row-standardized spatial weights matrix recording adjacencies
}

parameters {
  real constant; // beta_0 in Bill's model
  real effect; // effect of exposure on lip cancer
  real<lower=0> place_variance; // variance of the place-level random effect (a kind of residual)
  real place_re[n_obs]; // the place-level random effects
  real<lower=0> spillover_variance; // variance of the "spillover" effects on the next line
  real spillover_re[n_obs]; // spillover random effects, u_j^(3) in Bill's paper, Eq. 12
}

model {
  real log_lambda[n_obs]; // the mean of a poisson random variate
  place_variance ~ cauchy(0, 5); // prior for variance
  spillover_variance ~ cauchy(0, 5);
  place_re ~ normal(0, place_variance); // random effects for the place
  spillover_re ~ normal(0, spillover_variance); // random spillovers
  for (i in 1:n_obs) {
    log_lambda[i] = constant + effect*exposure[i] + place_re[i]; // mean of poisson is the exogenous info
    for (j in 1:n_obs) { 
      log_lambda[i] += spillover_re[j]*W[i,j]; // plus the spillovers
    }
    counts[i] ~ poisson(exp(log_lambda[i])); // and the outcome is distributed poisson
  }
}
