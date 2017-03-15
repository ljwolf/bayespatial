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
    """
    Adapted from pg. 63 of Anselin 1988, the logp for a SAR_error Model is
    -N/2(ln(pi * sigma^2)) + ln|A| \
    - .5*sigma^{-2}*(y - X\beta)'A'A(y-X\beta)

    where A is I - rho W

    Arguments
    ----------
    mu      :   mean of the distribution
    scale   :   univariate variance parameter
    rho     :   autoregressive parameter
    W       :   spatial weighting matrix
    """
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
        """

        Adapted from pg. 63 of Anselin 1988, the log likelihood for a lag model is:

        -N/2(ln(pi * scale^2)) + ln|A| \
        - .5*scale^{-2}*(Ay - mu)'(Ay-mu)

        where A is I - rho W


        Arguments
        ----------
        mu      :   mean of the distribution
        scale   :   univariate variance parameter
        rho     :   autoregressive parameter
        W       :   spatial weighting matrix
        """
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
        """

        Adapted from pg. 63 of Anselin 1988, the log likelihood for a combo model is:

        -N/2(ln(pi * scale^2)) + ln|A| + ln|B| \
        - .5*scale^{-2}*(Ay - mu)'B'B(Ay-mu)

        where B is I - lambda_ M and A is I - rho W


        Arguments
        ----------

        mu      :   mean of the distribution
        scale   :   univariate variance parameter
        rho     :   autoregressive parameter for the endogenous Lag component
        lambda_ :   autoregressive parameter for the spatial error component
        W       :   spatial weighting matrix for the endogenous Lag component
        M       :   spatial weighting matrix for the spatial error component, assumed to be W if not provided.
        """
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
        """
        the logp for a spatial moving average error structure is:
        
        -N/2 log(pi * sigma^2) - log(|A|) 
        - .5 * sigma^{-2} * (Y - X\beta)(AAt)^{-1}(Y - X\beta)

        where A is I + rho W

        Arguments
        ----------
        mu      :   mean of the distribution
        scale   :   univariate variance parameter
        rho     :   autoregressive parameter
        W       :   spatial weighting matrix  
        """
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
        the sparse cached log determinant assumes I - rho W = A, and computes
        the log determinant of A wrt rho with cached W. 

        To get this right with the SMA, we need to use -rho in the logdet. 
        """

        delta = value - self.mu
        ld = self.spld(-self.rho) 
        out = -self.W.n / 2.0 * tt.log(np.pi * self.scale)
        out -= ld

        kern = slinalg.solve(self.AAt, delta)
        kern = tt.mul(delta, kern)
        kern = kern.sum()
        kern = kern * self.scale**-2
        kern /= 2.0
        return out - kern

class CAR(mc.Continuous):
    def __init__(self, mu, scale, rho, W, *args, **kwargs):
        """
        the logp for a conditional autoregressive error structure is:
        
        -N/2 log(pi * sigma^2) - log(|A|) 
        - .5 * sigma^{-2} * (Y - X\beta)(A)(Y - X\beta)

        where A is D - rho W, and D is the degree matrix 
        for the graph contained in W

        Arguments
        ----------

        mu      :   mean of the distribution
        scale   :   univariate variance parameter
        rho     :   autoregressive parameter
        W       :   spatial weighting matrix  
        """
        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(mu)
        self.scale = scale
        self.W.transform = 'b'
        self.D = np.diag(W.sparse.toarray().sum(axis=1))
        self.Tau = (D - rho * W.sparse.toarray()) * scale **-2
        self.spld = CachedLogDet(W)

    def random(self, point=None, size=None):
        raise NotImplementedError
        mu, cov = draw_values([self.mu, self.cov], point=point)
        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(mean, cov, 
                                                 None if size==mean.shape else size)
        samples = generate_samples(_random, mean=mu, cov=cov, 
                                   dist_shape=self.shape, broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, value):
        delta = value - self.mu
        #CAR has no squared structure in the covariance, so it retains |Omega|^{-1/2}
        ld = self.spld(self.rho) * .5
        out = - self.W.n / 2.0 * tt.log(np.pi * self.scale)
        out -= ld

        kern = tt.dot(self.Tau, delta)
        kern = tt.mul(delta, kern).sum()
        kern *= self.scale **-2
        kern /= 2.0
        return out - kern