# pymc3_spatial
distributions for sparse spatial models in PyMC3. This is **pre-alpha**, and done to explore how to efficiently provide common spatial econometric/statistical specifications in PyMC3 distributions, like the exploration by for STAN [Max Josef](https://github.com/mbjoseph/CARstan). Notably, this exploration [avoids using precomputing the eigendecomposition of the weights matrix](http://econweb.umd.edu/~kelejian/Research/P071897.PDF), and instead opts for direct computation of the log determinant using a sparse matrix approach.

Spatial models, such as conditional and simultaneous autoregressive models, often scale poorly as the size of the problem increases. In general, this applies to any model with a multivariate normal response that *cannot be conditioned* into a case where the response is independent. This conditional independence is a very useful property for spatial multilevel models, but is unavailable when using an explicit model of spatial dependence in the covariance or response. In specifications where spatial dependence is modeled explicitly in the covariance matrix, the log determinant of the full covariance matrix is required, and is the [heaviest computational load for the model](https://brage.bibsys.no/xmlui/handle/11250/276920).

Correlation between observations that cannot be conditioned out means that the covariance matrix for the resulting model is not diagonal. Often, though, the covariance is incredibly sparse. Common spatial correlation structures operate under the warrant that near things are more related than far things, so most responses should be nearly uncorrelated with one another. Thus, when the log determinant of the covariance matrix is required in a multivariate normal model, a sparse matrix algorithm to compute the log determinant may be much faster than a dense matrix algorithm.

So, this repo contains a few models that aim to simplify spatial statistical models in pymc3. The five currently-supported distributions capture many common primitives in spatial modeling of lattice data, and should be composable like standard pymc3 distributions. Using a spatial linking matrix `W` and autoregressive parameter `rho`:
- Spatial Moving Average, a multivariate Normal with covariance `(I + rho W)(I + rho W) * sigma**2'`
- Simultaneous Autoregressive Error, a multivariate Normal with **precision** `(I - rho W)'(I - rho W) tau**2`
- Simultaneous Autoregressive Lag, a non-standard distribution with logp: `-N/2 log(pi sigma**2) + log|(I - rho W)| - .5 * sigma**2 * (Y - rho W Y - X beta)'(Y - rho W Y - X beta)`
- Simultaneous Combo Model, a mixture of the Autoregressive lag model and the autoregressive Error model. Letting `lambda_` be the autoregressive effect for the error component and `M` be the spatial linking matrix for that component, the logp of the combo model is `-N/2 log(pi sigma**2) + log(|I - rho W|) + log(|I - lambda_ M|) - .5 * sigma**-2 * (Y - rho W Y - X beta)'(I - lambda_ M)'(I - lambda_ M)(Y - rho W Y - X beta)`
- Conditional Autoregressive Model, a multivariate Normal with **precision** `(D - rho W) * tau**2`

With the exception of the lag and combo distributions, the remaining distributions are variants of the multivariate Normal implemented in PyMC3, but using a sparse log determinant `Op`, implemented in theano. The `Op`, called `CachedLogDet`, takes a `PySAL` weights object at initialization:
```
... 
class CachedLogDet(Op):
    def __init__(self, W):
        """
        Initialize the log determinant of I - rho W
        """
        self.Is = spar.csc_matrix(spar.identity(W.n))
        self.I = np.eye(W.n)
        self.Ws = W.sparse
        self.W = W.sparse.toarray()
...
spld = CachedLogDet(W)
```
So, each distribution used in a model stores its own sparse log determinant node. The resulting node takes a scalar input `rho` and performs a log determinant *via* sparse LU decomposition. This is a similar strategy to the new PyMC3 LogAbsDet op, but uses Sparse LU decomposition instead of dense SVD. Since the covariance is positive semidefinite, the sum of the log absolute value of the relevant LU matrix diagonal is the log determinant of the target matrix:
```python
from scipy.sparse.linalg import import splu
from scipy.sparse import identity as speye

...
def perform(self, inputs, outputs):
    (rho, ) = inputs
    (ld, ) = outputs
    A = self.Is - rho * self.Ws #cached sparse W and Identity during the init
    Ud = splu(A).U.diagonal()
    ld[0] = np.sum(np.log(np.abs(Ud)))
```
And, the gradient is the trace of `A^{-1} W`. This is implemented using `theano.tensor.slinalg.solve(A, W)`, operating on `A` as a dense matrix. A sparse solve op might make this faster, if `spsolve` has any speed gains over a dense `solve`. 