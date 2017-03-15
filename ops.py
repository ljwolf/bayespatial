import scipy.sparse as spar
import scipy.sparse.linalg as spla
import theano.tensor as tt
from theano.tensor import slinalg, nlinalg
import theano.sparse as ts
from theano.gof import Apply
from theano import Op
import numpy as np

import theano as th
th.config.exception_verbosity='high'

# define this as if it's in terms of rho and W, and give the derivatives in
# terms of rho, since that's what the graph is expecting
"""
class Sparse_LapDet(Op):
    Sparse Matrix Determinant of a Laplacian Matrix using Sparse LU
    Decomposition

    def make_node(self, rho, W):
        self.A = ts.as_sparse(spar.eye(W.n) - rho * )
        ld = tt.scalar(dtype=rho.dtype)
        return Apply(self, [rho, W], [ld])

    def make_node_(self, A):
        A = ts.as_sparse(A)
        ld = tt.scalar(dtype=A.dtype)
        return Apply(self, [A], [ld])
    
    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (z, ) = outputs
        Ud = spla.splu(A).U.diagonal()
        ld = np.asarray(np.sum(np.log(np.abs(Ud))))
        z[0] = ld

    def perform_from_bits(self, node, inputs, outputs):
        (rho, W,) = inputs
        (z, ) = outputs
        As = spar.to_csc(spar.eye(W.n)) - ts.mul(rho, W.sparse)
        Ud = spla.splu(As).U.diagonal()
        ld = np.asarray(np.sum(np.log(np.abs(Ud))))
        z[0] = ld

    def grad(self, inputs, g_outputs):
        (A, ) = inputs
        (gz,) = g_outputs
        Ad = A.toarray()
        dinv = tt.nlinalg.matrix_inverse(Ad).T
        out = tt.mul(dinv, - Ad)
        return [ts.csc_from_dense(tt.mul(out, gz))]

    def grad_from_bits(self, node, inputs, outputs):
        (rho, W,) = inputs
        (gz, ) = outputs
        As = spar.to_csc(spar.eye(W.n)) - ts.mul(rho, W.sparse)
"""
class Dense_LULogDet(Op):
    """Log Determinant of a matrix by sparse LU decomposition,
       from dense inputs. Use when casting has no significant overhead."""
    def make_node(self, A):
        A = tt.as_tensor(A)
        ld = tt.scalar(dtype=A.dtype)
        return Apply(self, [A], [ld]) 
    
    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (z,) = outputs
        As = spar.csc_matrix(A)
        Ud = spla.splu(As).U.diagonal()
        ld = np.sum(np.log(np.abs(Ud)))
        z[0] = ld

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [A] = inputs
        dinv = nlinalg.matrix_inverse(A).T
        dout = tt.dot(gz, dinv)
        return [dout]

class Dense_LogDet(Op):
    """Log Determinant of a matrix using numpy.linalg.slogdet.
       Use as a reference implementation"""
    def make_node(self, A):
        A = tt.as_tensor(A)
        ld = tt.scalar(dtype=A.dtype)
        return Apply(self, [A], [ld]) 
    
    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (z,) = outputs
        sgn, ld = np.linalg.slogdet(A)
        if sgn not in [-1,0,1]:
            raise Exception('Loss of precision in log determinant')
        ld *= sgn
        z[0] = ld

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [A] = inputs
        dinv = nlinalg.matrix_inverse(A).T
        dout = tt.dot(gz, dinv)
        return [dout]

class SparseLogDet(Op):
    def make_node(self, x):
        x = ts.as_sparse_variable(x)
        out = tt.scalar(dtype=x.dtype)
        return Apply(self, [x], [out])
    
    def perform(self, node, inputs, outputs, params=None):
        (x, ) = inputs
        (z, ) = outputs

        U = spla.splu(x).U
        ld = np.sum(np.log(np.abs(U.diagonal())))
        z[0] = np.asarray(ld, dtype=x.dtype)

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [x] = inputs
        return [gz * tt.nlinalg.matrix_inverse(x.toarray()).T]

class CachedLogDet(Op):
    def __init__(self, W):
        """
        Initialize the log determinant of I - rho W
        """
        self.Is = spar.csc_matrix(spar.identity(W.n))
        self.I = np.eye(W.n)
        self.Ws = W.sparse
        self.W = W.sparse.toarray()

    def make_node(self, rho):
        rho = tt.as_tensor(rho)
        ld = tt.scalar(dtype=rho.dtype)
        return Apply(self, [rho], [ld])

    def perform(self, node, inputs, outputs, params=None):
        (rho, ) = inputs
        (ld, ) = outputs

        As = spar.csc_matrix(self.Is - rho * self.Ws)
        U = spla.splu(As).U
        val = np.sum(np.log(np.abs(U.diagonal())))
        ld[0] = np.asarray(val, dtype=rho.dtype)

    def grad(self, inputs, g_outputs):
        # let A = I - rho W, and dRho(A) be the derivatrive of A wrt Rho
        # dRho(log(|(AtA)|)) = dRho(log(|(AtA)|)))
        # = dRho(log(|At|) + log(|A|))
        # = dRho(log(|A| + log(|A|)))
        # = 2 dRho(log(|A|))
        # = 2 |A|^{-1} dRho(|A|) = 2|A|^{-1} tr(Adj(A)dRho(A))
        # = 2 |A|^{-1} |A| tr(A^{-1}(-W)) = 2 * tr(A^{-1}W)
        [gz] = g_outputs
        [rho] = inputs
        A = self.I - rho * self.W
        #trAiW = slinalg.solve(A, self.W).diagonal().sum()
        trAiW = (nlinalg.matrix_inverse(A).dot(self.W)).diagonal().sum()
        return [trAiW]