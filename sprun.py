import pysal as ps
import scipy.sparse as spar
import theano.tensor as tt
import theano
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
import theano.sparse as ts
import numpy as np
import pymc3 as mc
from distributions import SAR_Error


data = ps.pdio.read_files(ps.examples.get_path('south.shp'))
data = data.query('STATE_NAME in ("Texas", "Louisiana", '
                  '"Oklahoma", "Arkansas")')
W = ps.weights.Queen.from_dataframe(data)
W.transform = 'r'

Y = data[['HR90']].values
X = data[['GI89', 'BLK90']].values

N, P = X.shape

I = spar.csc_matrix(spar.eye(W.n))

with mc.Model() as SE:

    a = mc.Normal('alpha', 0, 1)
    beta = mc.Normal('beta', 0, 1, shape=P)

    sigma = mc.HalfCauchy('sigma', 5, testval=2)
    #bnorm = mc.Bound(mc.Normal, lower=-.95, upper=.95)
    #rho = bnorm('rho', mu=0, sd=.1, testval=0)
    rho = mc.Uniform('rho', lower=-.99, upper=.99, testval=0)
    #Ad = np.eye(W.n) - rho * W.sparse.toarray()
    #As = I - ts.mul(rho, W.sparse)

    #AtAs = ts.structured_dot(ts.transpose(As), As)
    #AtA = ts.dense_from_sparse(AtAs)
    #tau = tt.mul(sigma**-2, AtA)
    #taus = ts.mul(sigma**-2, AtAs)

    # transpose has a regular gradient
    #AtA = tt.dot(tt.transpose(Ad), Ad)
    #AtAs = ts.csc_from_dense(AtA)
    #taus = ts.mul(sigma**-2, AtAs)
    #tau = tt.mul(sigma**-2, AtA)

    mu = a + tt.dot(X, beta)

    #cached_splogdet = CachedLogDet(W)

    #def se_logp(value):
    #    delta = value - mu
    #    ld = cached_splogdet(rho)
    #    ld = ld * 2 * sigma
    #    out = -N / 2.0
    #    out -= ld

    #    kern = tt.dot(AtA, delta)
        #kern = tt.dot(ts.dense_from_sparse(taus), delta) # shape issue in MulSD.grad
        #kern = ts.dot(taus, delta) # AssertionError in _is_sparse_variable(gz) in MulSD.grad
    #    kern = tt.mul(delta, kern)
    #    kern = kern * sigma**-2
    #    kern = kern.sum()
    #    #kern /= 2.0
    #    return out - kern

    #outcome = mc.MvNormal('outcome', mu=mu,
    #                      tau=tau,
    #                      observed=Y.flatten(), shape=N)
    outcome = SAR_Error('outcome', mu=mu, scale=sigma, rho=rho, 
                        W=W, observed=Y.flatten(), shape=N)
    #outcome = mc.DensityDist('outcome', logp=se_logp, observed=Y)

    #start = mc.find_MAP()
    #step = mc.NUTS(start=start)
    trace = mc.sample(50)
    #trace = mc.sample(500, step=mc.Metropolis())
