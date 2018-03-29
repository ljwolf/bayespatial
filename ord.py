import theano.tensor as tt
import theano.sparse as ts
import theano
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
import numpy as np
import pymc3 as mc
from pymc3.distributions.dist_math import bound as dist_bound
from theano.tensor import slinalg as tslinalg
from theano.tensor import nlinalg as tnlinalg
import scipy.sparse as sparse
from scipy import linalg as slinalg
#from libpysal.api import W
from pysal import W

class SAR_Error(mc.Continuous):
    def __init__(self, mean, scale, autoreg, weights, 
                 eigs=None, lower=None, upper=None, 
                 *args, **kwargs):
        dense_adj, sparse_adj = _resolve_weights(weights)
        self.n = dense_adj.shape[0]
        kwargs['shape'] = self.n
        super(SAR_Error, self).__init__(*args, **kwargs)
        self.mean = tt.as_tensor_variable(mean)
        self.scale = tt.as_tensor_variable(scale)
        self.autoreg = tt.as_tensor_variable(autoreg)
        #self.spweights = ts.as_sparse_variable(sparse_adj)
        self.weights = dense_adj 
        #self.laplacian = tt.eye(self.n) - self.spweights * self.autoreg
        self.laplacian = tt.eye(self.n) - self.weights * self.autoreg
        #self.AtA = tt.dot(self.laplacian.T, self.laplacian)
        self.eigs = slinalg.eigvalsh(dense_adj) if eigs is None else eigs
        self.lower = 1/self.eigs.min() if lower is None else lower
        self.upper = 1/self.eigs.max() if upper is None else upper
        assert len(self.eigs.shape)== 1
        assert len(self.eigs) == self.n
        self.const_factor = -self.n * tt.log(np.pi * self.scale) * .5
    
    def logp(self, value):
        
        delta = value - self.mean # Y - Xbeta
        dfilt = tt.dot(self.laplacian, delta) # (I - rho W)*(Y - Xbeta)
        kernel = tt.dot(dfilt, dfilt.T).trace() # tr((y - Xbeta)(I - rho W)[(y - Xbeta)(I - rho W)]')
        #kernel = tt.dot(delta.T, tt.dot(self.AtA, delta))
        kernel = kernel / (2 * self.scale ** 2) # tr(delta A (delta A)')
        logdet = tt.log(1 - self.autoreg * self.eigs).sum() # sum(log(1 - rho eigvals))
        
        # -n/2 * log(pi * sigma) + logdet - kernel
        
        return dist_bound(self.const_factor + logdet - kernel, 
                          self.autoreg < self.upper, self.autoreg>self.lower)
    
class SAR_Lag(mc.Continuous):
    def __init__(self, mean, scale, autoreg, weights, 
                 eigs=None, lower=None, upper=None, 
                 *args, **kwargs):
        dense_adj, sparse_adj = _resolve_weights(weights)
        self.n = dense_adj.shape[0]
        kwargs['shape'] = self.n
        super(SAR_Lag, self).__init__(*args, **kwargs)
        self.mean = tt.as_tensor_variable(mean)
        self.scale = tt.as_tensor_variable(scale)
        self.autoreg = tt.as_tensor_variable(autoreg)
        #self.spweights = ts.as_sparse_variable(sparse_adj)
        self.weights = dense_adj 
        #self.laplacian = tt.eye(self.n) - self.spweights * self.autoreg
        self.laplacian = tt.eye(self.n) - self.weights * self.autoreg
        #self.AtA = tt.dot(self.laplacian.T, self.laplacian)
        self.eigs = slinalg.eigvalsh(dense_adj) if eigs is None else eigs
        self.lower = 1/self.eigs.min() if lower is None else lower
        self.upper = 1/self.eigs.max() if upper is None else upper
        assert len(self.eigs.shape)== 1
        assert len(self.eigs) == self.n
        self.const_factor = -self.n * tt.log(np.pi * self.scale) * .5
    
    def logp(self, value):
        
        Ay = tt.dot(self.laplacian, value)
        delta = Ay - self.mean
        dfilt = tt.dot(delta.T, delta) # ((I - rho W)*Y - Xbeta)
        kernel = kernel / (2 * self.scale ** 2) # 
        logdet = tt.log(1 - self.autoreg * self.eigs).sum() # sum(log(1 - rho eigvals))
        
        # -n/2 * log(pi * sigma) + logdet - kernel
        
        return dist_bound(self.const_factor + logdet - kernel, 
                          self.autoreg < self.upper, self.autoreg>self.lower)

class SAR_Combo(mc.Continuous):
    def __init__(self, mean, scale, endog_autoreg, error_autoreg,
                 weights=None, endog_weights=None, error_weights=None, 
                 error_eigs=None, endog_eigs=None,
                 *args, **kwargs):
        if weights is None:
            assert endog_weights is not None, 'Either one `weights` argument or both '\
                                              'endog_weights and error_weights must be'\
                                              ' provided.'
            assert error_weights is not None, 'Either one `weights` argument or both '\
                                              'endog_weights and error_weights must be'\
                                              ' provided.'
            error_dense_adj, error_sparse_adj = _resolve_weights(error_weights)
            endog_dense_adj, endog_sparse_adj = _resolve_weights(enog_weights)
            self.n = error_dense_adj.shape[0]

            self.error_eigs = slinalg.eigvalsh(error_dense_adj) if error_eigs is None else eigs
            self.endog_eigs = slinalg.eigvalsh(endog_dense_adj) if endog_eigs is None else eigs
        else:
            assert endog_weights is None, 'Either one `weights` argument or both '\
                                          'endog_weights and error_weights must be'\
                                          ' provided.'
            assert endog_weights is None, 'Either one `weights` argument or both '\
                                          'endog_weights and error_weights must be'\
                                          ' provided.'
            dense_adj, sparse_adj = _resolve_weights(weights)
            self.error_weights = dense_adj
            self.endog_weights = dense_adj
            self.error_eigs = self.endog_eigs = slinalg.eigvalsh(weights)
            
            self.n = dense_adj.shape[0]
        kwargs['shape'] = self.n
        super(SAR_Lag, self).__init__(*args, **kwargs)
        self.mean = tt.as_tensor_variable(mean)
        self.scale = tt.as_tensor_variable(scale)
        self.error_autoreg = tt.as_tensor_variable(error_autoreg)
        self.endog_autoreg = tt.as_tensor_variable(endog_autoreg)
        #self.spweights = ts.as_sparse_variable(sparse_adj)
        eye = tt.eye(self.n)
        self.error_laplacian = eye - self.error_weights * self.error_autoreg
        self.endog_laplacian = eye - self.endog_weights * self.endog_autoreg
        self.error_lower = 1/self.error_eigs.min()
        self.error_upper = 1/self.error_eigs.max()
        self.endog_lower = 1/self.endog_eigs.min()
        self.endog_upper = 1/self.endog_eigs.max()
        assert len(self.error_eigs.shape) == len(self.endog_eigs.shape) == 1
        assert len(self.error_eigs) == len(self.endog_eighs) ==  self.n
        self.const_factor = -self.n * tt.log(np.pi * self.scale) * .5
        
    def logp(self, value):
        
        delta = tt.dot(self.endog_laplacian, value) - self.mean
        
        delta_B = tt.dot(delta, self.endog_laplacian)
        
        kernel = tt.dot(delta_B, delta_B.T).trace() / (2 * self.scale ** 2)
        
        error_logdet = tt.log(1 - self.error_autoreg * self.error_eigs).sum()
        endog_logdet = tt.log(1 - self.endog_autoreg * self.endog_eigs).sum()

        return dist_bound(dist_bound(self.const_factor + error_logdet\
                                     + endog_logdet - kernel,
                                     self.error_autoreg > self.error_lower, 
                                     self.error_autoreg < self.error_upper),
                          self.endog_autoreg > self.endog_lower, 
                          self.endog_autoreg < self.endog_upper)

class SMA(mc.Continuous):
    def __init__(self, mean, scale, autoreg, weights, eigs,
                 *args, **kwargs):
        dense_adj, sparse_adj = _resolve_weights(weights)
            
        self.n = dense_adj.shape[0]
        kwargs['shape'] = self.n
        super(SAR_Lag, self).__init__(*args, **kwargs)
        self.mean = tt.as_tensor_variable(mean)
        self.scale = tt.as_tensor_variable(scale)
        self.autoreg = tt.as_tensor_variable(autoreg)
        self.weights = dense_adj
        eye = tt.eye(self.n)
        self.half_macov = eye + self.weights * self.autoreg
        self.lower = -1/self.error_eigs.max()
        self.upper = -1/self.error_eigs.min()
        assert len(self.eigs.shape) == 1
        assert len(self.eigs) == self.n
        self.const_factor = -self.n * tt.log(np.pi * self.scale) * .5
        
    def logp(self, value):
        
        delta = value - self.mean
                
        kernel = tt.dot(delta.T, self.half_macov).dot(self.half_macov.T, delta)
        kernel = kernel / (2 * self.scale ** 2)
        
        logdet = tt.log(1 + self.autoreg * self.eigs).sum()

        return dist_bound(self.const_factor + logdet - kernel,
                          self.autoreg > self.lower, 
                          self.autoreg < self.upper) 
        
def _resolve_weights(weights):
    if sparse.issparse(weights):
        weights = weights / weights.sum(axis=1)
        dense_adj = weights.toarray()
        sparse_adj = weights
    elif isinstance(weights, np.ndarray):
        weights = weights / weights.sum(axis=1)
        dense_adj = weights
        sparse_adj = spar.csc_matrix(weights)
    elif isinstance(weights, W):
        weights.transform = 'r'
        sparse_adj = weights.sparse
        dense_adj = weights.sparse.toarray()
    else:
        raise TypeError('Type of weights passed to distribution not understood.')
    return dense_adj, sparse_adj
