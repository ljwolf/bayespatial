functions{

  real sparse_sar_error_lpdf(vector Y, vector mu, real tau, matrix AtA,
                             real ldet, matrix eye, int n){
    vector[n] ldets ;
    real kern;
    kern = quad_form(AtA, Y - mu);

    return .5 * (n * log(3.141^.5 * tau)) + ldet - .5 * kern;
  }

  real sparse_sar_lag_lpdf(vector Y, vector mu, real tau, matrix A,
                          real ldet, matrix eye, int n){
    real kern;
    kern = quad_form(crossprod(A), A*Y - mu);

    return -.5 * (n * log(3.141^.5 * tau)) + ldet - .5*kern;
  }

}
data {
    int<lower = 1> n;
    int<lower = 1> p;
    matrix[n,p] X;
    vector[n] y;
    matrix<lower =0, upper = 1>[n,n] W;
    vector[n] evalues;
}

transformed data{
    vector[n] zeros;
    matrix[n,n] eye;
    eye = diag_matrix(rep_vector(1.0, n));
/*
    vector[n] evalues;
    evalues = eigenvalues_sym(W);
*/
  }
parameters{
  real constant;
  vector[p] beta;
  real<lower=0>tau;
  real<lower=1/min(evalues), upper=1> lambda;
}
transformed parameters{
  matrix[n,n] A;
  matrix[n,n] AtA;
  real ldet;
  A = (eye - lambda * W);
  AtA = crossprod(A);
  ldet = 0;
  for (i in 1:n){
    ldet = ldet + log1m(lambda * evalues[i]);
  }
}

model{
  constant ~ normal(0,1);
  lambda ~ normal(0,.1);
  beta ~ normal(0,1);
  tau ~ cauchy(0, 5);
  /*y ~ sparse_sar_error(X * beta + constant, tau, AtA, ldet, eye, n);*/
  /*y ~ sparse_sar_lag(X * beta + constant, tau, A, ldet, eye, n);*/
  /*y ~ normal(X * beta + constant, tau);
}
