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
        trAiW = slinalg.solve(A, self.W).diagonal().sum()
        #trAiW = (nlinalg.matrix_inverse(A).dot(self.W)).diagonal().sum()
        return [trAiW]

    def spgrad(self, inputs, g_outputs):
        raise NotImplementedError

class OrdLogDet(Op):
    def __init__(self, W):
        """
        Initialize the log determinant of I - rho W
        """
        self.I = np.eye(W.n)
        self.W = W.sparse.toarray()
        self.evals = np.linalg.eigvals(self.W)

    def make_node(self, rho):
        rho = tt.as_tensor(rho)
        ld = tt.scalar(dtype=rho.dtype)
        return Apply(self, [rho], [ld])

    def perform(self, node, inputs, outputs, params=None):
        (rho, ) = inputs
        (ld, ) = outputs

        ld[0] = np.asarray(np.sum(np.log(1 - self.evals * rho)), dtype=rho.dtype)

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
        trAiW = slinalg.solve(A, self.W).diagonal().sum()
        #trAiW = (nlinalg.matrix_inverse(A).dot(self.W)).diagonal().sum()
        return [trAiW]

    def spgrad(self, inputs, g_outputs):
        raise NotImplementedError


class SpSolve(Op):
    def make_node(self, A, b):
        A = ts.as_sparse_variable(A)
        b = ts.as_sparse_or_tensor_variable(b)
        assert A.ndim == 2
        assert b.ndim in [1,2]

        x = tt.tensor(dtype=b.dtype)
        return Apply(self, [A,b], [x])

    def perform(self, node, inputs, output):
        A,b = inputs

        rval = spla.spsolve(A,b)
        output[0][0] = rval

    def grad(self, input, output_gradients):
        A,b = inputs
        c = self(A,b)
        c_bar = output_gradients[0]
        trans_solve_op = SpSolve()
        b_bar = trans_solve_op(ts.transpose(A), c_bar)
        A_bar = -ts.outer(b_bar, c) if c.ndim == 1 else -b_bar.dot(c.T)
        return [A_bar, b_bar]