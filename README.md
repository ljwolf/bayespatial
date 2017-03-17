# bayespatial
This is a **pre-alpha** exploration of whether linear models with spatial effects can be done *efficiently* in PyMC3 & stan. My target is to make the sampling as efficient & fast as what I've implemented in Gibbs samplers in [spvcm](https://github.com/ljwolf/spvcm), which models two-level spatially-correlated variance components models using `numpy`.

This incorporates a bit of what I've been doing trying to make PyMC3 & Stan work well with sparse covariance matrices in SAR specifications, as well as the special endogenous lag logp functions. This exploration compares using eigenvalue factorization due to Ord (1975), Sparse Matrix Factorization from SuperLU, and the existing dense Singular Value Decomposition op used in PyMC3 for the log determinant. For a comparison of the speed of the various log determinant methods, check `looking_at_ops.ipynb`. For an example on how to use these methods in a model in PyMC3, look at `example_modelfit.ipynb`. Stan ops for the mixed regressive autoregressive model and SAR-error model are contained in the `stan` branch.

Both implementations appear to be "correct," but gradient sampling is exceedingly slow in PyMC3, even with ADVI initialization for NUTS. Stan is also slower than I'd like. 

I know the gradient with respect to the spatial correlation parameter (of any specification) involves a trace of the form `tr((I - rho W)^{-1}W)`, where `W` is `n x n`. That seems to be the main constraint on scaling the gradient sampling. 

What works:
- Ord (1975) Eigenvalue log determinant op with gradient. Performs: `log |I - rho W| = sum(1 - rho * eigvals)`
- Sparse LU factorization op with gradient. Performs: `log |I - rho W| = |L||U| = sum(log(|diag(U)|))`
- Models with Gaussian SAR and spatial moving average terms or responses

## Why is spatial special?
Spatial models, such as conditional and simultaneous autoregressive models, often scale poorly as the size of the problem increases. In general, this applies to any model with a multivariate normal response that *cannot be conditioned* into a independence. This conditional independence is a very useful property for spatial multilevel models with exchangeable groups, where conditioning on the upper-level processes provides independence in the lower level processes. But, this is unavailable when using an explicit model of spatial dependence in the covariance at a given level. In specifications where spatial dependence is modeled explicitly in the covariance matrix, the log determinant of the full covariance matrix is required, and is the [heaviest computational load for the model](https://brage.bibsys.no/xmlui/handle/11250/276920).

Often, though, the covariance is incredibly sparse. Common spatial correlation structures rely on the idea that near things are more related than far things, so most responses should be nearly uncorrelated with one another. Thus, when the log determinant of the covariance matrix is required in a multivariate normal distribution, a sparse matrix algorithm to compute the log determinant may be much faster than a dense matrix algorithm, since the vast majority of the entries in the covariance matrix *are* zero.

So, this repo contains a few models & ops for spatial statistical models in PyMC3. Four distributions capture many common primitives in spatial modeling of lattice data using simultaneously-autoregressive effects. The distributions should be composable like standard PyMC3 distributions. Using a spatial linking matrix `W` and autoregressive parameter `rho`:
- Spatial Moving Average, a multivariate Normal with covariance `(I + rho W)(I + rho W) * sigma**2'`
- Simultaneous Autoregressive Error, a multivariate Normal with **precision** `(I - rho W)'(I - rho W) tau**2`
- Simultaneous Autoregressive Lag, a non-standard distribution with logp: `-N/2 log(pi sigma**2) + log|(I - rho W)| - .5 * sigma**2 * (Y - rho W Y - X beta)'(Y - rho W Y - X beta)`
- Simultaneous Combo Model, a mixture of the Autoregressive lag model and the autoregressive Error model. Letting `lambda_` be the autoregressive effect for the error component and `M` be the spatial linking matrix for that component, the logp of the combo model is `-N/2 log(pi sigma**2) + log(|I - rho W|) + log(|I - lambda_ M|) - .5 * sigma**-2 * (Y - rho W Y - X beta)'(I - lambda_ M)'(I - lambda_ M)(Y - rho W Y - X beta)`
Doing an efficient sparse CAR model may require different sparse ops, but it should be possible to implement this using the same general approach.

The SAR-Error and Spatial Moving Average models are essentially the standard `MvNormal` implemented in PyMC3, but use a special method to compute the logp efficiently. The Combo and Lag models are also similar to the typical `PyMC3` normal distributions, but use different means. Regardless, all models use a sparse log determinant `Op`, implemented in theano. The comparison of these methods is done in `looking_at_ops.ipynb`.
