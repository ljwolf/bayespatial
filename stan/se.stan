functions{
  real sparse_sar_error_lpdf(vector Y, vector mu, real tau, real lambda,
                             matrix W, vector evals, matrix eye, int n){
    vector[n] ldets ;
    matrix[n,n] A;
    real kern;
    A = (eye - lambda * W);
    kern = quad_form_sym(crossprod(A), Y - mu);

    for (i in 1:n){ 
      ldets[i] = log1m(lambda * evals[i]);
    }

    return .5 * (n * log(3.141^.5 * tau)) + sum(ldets) - .5 * kern;
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
  vector[p] beta;
  real<lower=0>tau;
  real<lower=1/min(evalues), upper=1/max(evalues)> lambda;
}

model{
  lambda ~ uniform(1/min(evalues), 1/max(evalues));
  beta ~ normal(0,1);
  tau ~ cauchy(0, 5);
  y ~ sparse_sar_error(X * beta, tau, lambda, W, evalues, eye, n);
}