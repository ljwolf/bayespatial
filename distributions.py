import theano.tensor as tt
import theano
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
import numpy as np
import pymc3 as mc
from ops import CachedLogDet
from theano.tensor import slinalg

from scipy import stats

def get_dense_cov(mu, tau):
    cov = tt.nlinalg.matrix_inverse(tau)
    return cov

from pymc3.distributions.distribution import draw_values, generate_samples

class SAR_Error(mc.Continuous):
    def __init__(self, mu, scale, rho, W,
                 *args, **kwargs):
        super(SAR_Error, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu= tt.as_tensor_variable(mu)
        self.scale = scale
        self.W = W
        self.Ws = W.sparse
        self.spld = CachedLogDet(self.W)
        self.rho = rho
        self.A = np.eye(W.n) - rho * W.sparse.toarray()
        self.AtA = tt.dot(tt.transpose(self.A), self.A)
        self.tau = tt.mul(self.AtA, self.scale **-2)

        cov = get_dense_cov(self.mu, self.tau)
        self.cov = tt.as_tensor_variable(cov)

    def random(self, point=None, size=None):
        mu, cov = draw_values([self.mu, self.cov], point=point)
        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(mean, cov, 
                                                 None if size==mean.shape else size)
        samples = generate_samples(_random, mean=mu, cov=cov, 
                                   dist_shape=self.shape, broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, value):
        """
        Adapted from pg. 63 of Anselin 1988, the logp for a SAR_error Model is
        -N/2(ln(pi * sigma^2)) + ln|A| \
        - .5*sigma^{-2}*(y - X\beta)'A'A(y-X\beta)

        where A is I - rho W
        """

        delta = value - self.mu
        ld = self.spld(self.rho)
        out = -self.W.n / 2.0 * tt.log(np.pi * self.scale)
        out += ld

        kern = tt.dot(self.AtA, delta)
        #kern = tt.dot(ts.dense_from_sparse(taus), delta) # shape issue in MulSD.grad
        #kern = ts.dot(taus, delta) # AssertionError in _is_sparse_variable(gz) in MulSD.grad
        kern = tt.mul(delta, kern)
        kern = kern * self.scale**-2
        kern = kern.sum()
        kern /= 2.0
        return out - kern

class SAR_Lag(mc.Continuous):
    def __init__(self, mu, scale, rho, W,
                 *args, **kwargs):
        super(SAR_Error, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu= tt.as_tensor_variable(mu)
        self.scale = scale
        self.W = W
        self.Ws = W.sparse
        self.spld = CachedLogDet(self.W)
        self.rho = rho
        self.A = np.eye(W.n) - rho * W.sparse.toarray()

    def random(self, point=None, size=None):
        raise NotImplementedError
        # This needs to give Ai(X\beta + \epsilon)
        mu, cov = draw_values([self.mu, self.cov], point=point)
        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(mean, cov, 
                                                 None if size==mean.shape else size)
        samples = generate_samples(_random, mean=mu, cov=cov, 
                                   dist_shape=self.shape, broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, value):
        """
        Adapted from pg. 63 of Anselin 1988, the logp for a SAR-Lag variable is:
        -N/2(ln(pi * sigma^2)) + ln|A|
        - .5*sigma^{-2}*(Ay - X\beta)'(Ay-X\beta)

        where A is I - rho W
        """

        delta = tt.dot(self.A, value) - self.mu
        ld = self.spld(self.rho)
        out = -self.W.n / 2.0 * tt.log(np.pi * self.scale)
        out += ld

        kern = tt.mul(delta, delta).sum() * self.scale**-2
        kern *= .5
        return out - ker

class SAR_Combo(mc.Continuous):
    def __init__(self, mu, scale, rho, lambda_, W, M=None,
                 *args, **kwargs):
        super(SAR_Combo, self).__init__(*args, **kwargs)
        if M is None:
            M = W
        self.mean = self.median = self.mode = self.mu = mu= tt.as_tensor_variable(mu)
        self.scale = scale
        self.W = W
        self.Ws = W.sparse
        self.M = M
        self.Ms = M.sparse
        self.spldW = CachedLogDet(self.W)
        self.spldM = CachedLogDet(self.M)
        self.rho = rho
        self.lambda_ = lambda_
        self.A = np.eye(W.n) - rho * W.sparse.toarray()
        self.B = np.eye(W.n) - lambda_ * M.sparse.toarray()
        self.BtB = tt.dot(tt.transpose(self.B), self.B)

    def random(self, point=None, size=None):
        raise NotImplementedError
        # This needs to give Ai(X\beta + (Bi)\epsilon)
        mu, cov = draw_values([self.mu, self.cov], point=point)
        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(mean, cov, 
                                                 None if size==mean.shape else size)
        samples = generate_samples(_random, mean=mu, cov=cov, 
                                   dist_shape=self.shape, broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, value):
        """
        Adapted from pg. 63 of Anselin 1988, the log likelihood for a combo model is:

        -N/2(ln(pi * sigma^2)) + ln|A| + ln|B| \
        - .5*sigma^{-2}*(Ay - X\beta)'B'B(Ay-X\beta)

        where B is I - lambda W and A is I - rho W
        """
        delta = tt.dot(self.A, value) - self.mu
        ldA = self.spld_A(self.rho)
        ldB = self.spld_B(self.lambda_)
        out = -self.W.n / 2.0 * tt.log(np.pi * self.scale)
        out += ldA
        out += ldB

        kern = tt.dot(self.BtB, delta)
        kern = tt.mul(delta, kern).sum()
        kern *= self.scale**-2
        kern *= .5
        return out - ker

class SMA(mc.Continuous):
    def __init__(self, mu, scale, rho, W,
                 *args, **kwargs):
        super(SAR_Error, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu= tt.as_tensor_variable(mu)
        self.scale = scale
        self.W = W
        self.Ws = W.sparse
        self.spld = CachedLogDet(self.W)
        self.rho = rho
        self.A = np.eye(W.n) + rho * W.sparse.toarray()
        self.AAt = tt.dot(tt.transpose(self.A), self.A)
        self.cov = tt.mul(self.AAt, self.scale **2)

        self.cov = tt.as_tensor_variable(cov)

    def random(self, point=None, size=None):
        mu, cov = draw_values([self.mu, self.cov], point=point)
        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(mean, cov, 
                                                 None if size==mean.shape else size)
        samples = generate_samples(_random, mean=mu, cov=cov, 
                                   dist_shape=self.shape, broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, value):
        """
        the logp for a spatial moving average error structure is:
        
        -N/2 log(pi * sigma^2) - log(|A|) 
        - .5 * sigma^{-2} * (Y - X\beta)(AAt)^{-1}(Y - X\beta)

        where A is I + rho W

        """

        delta = value - self.mu
        ld = self.spld(self.rho)
        out = -self.W.n / 2.0 * tt.log(np.pi * self.scale)
        out -= ld

        kern = slinalg.solve(self.AAt, delta)
        kern = tt.mul(delta, kern)
        kern = kern * self.scale**-2
        kern = kern.sum()
        kern /= 2.0
        return out - kern