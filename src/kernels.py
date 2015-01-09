import numpy as np
import learn_predict
import scipy

import scipy.sparse.csr
def kernel_linear_unary(X_train, X_test, lhp, no_jitter):
    global dtype
    p = np.dot(X_train,X_test.T)
    if isinstance(p, scipy.sparse.csr.csr_matrix):
        p = p.toarray() # cos if using X_train sparse vector, p will be a csr_matrix -- incidentally in this case the resulting k_unary cannot be flattened, it will result in a (1,X) 2D matrix !
    k_unary = np.exp(lhp["unary"]) * np.array(p, dtype=learn_predict.dtype_for_arrays)
    return jitterize(k_unary, lhp, no_jitter)
    
import sklearn.metrics.pairwise
def kernel_exponential_unary(X_train, X_test, lhp, no_jitter):
    """
    $k_\text{exponential}(\mathbf{x},\mathbf{x'}) = \exp({-\frac{1}{2 \ell ^2} \Vert \mathbf{x} - \mathbf{x'} \Vert ^2})$  
    $k_\text{exponential ARD}(\mathbf{x},\mathbf{x'}) = \exp({-\frac{1}{2} \sum_i \frac{1}{\sigma_i ^2} ( \mathbf{x}_i - \mathbf{x'}_i ) ^2})$  
    """
    global dtype
    p = sklearn.metrics.pairwise.euclidean_distances(X_train, X_test, squared=True)
    # sames as scipy.spatial.distance.cdist(X_train,X_test, 'sqeuclidean')
    # but works with Scipy sparse and Numpy dense arrays
    # I thought it would be equal to X_train.dot(X_train.T) + X_test.dot(X_test.T) - X_train.dot(X_test.T) - X_test.dot(X_train.T))
    # but it doesnt seem to
    k_unary = learn_predict.dtype_for_arrays(np.exp(lhp["unary"]) * np.exp( -(1/2) * 1/(np.exp(lhp["length_scale"])**2 ) * p))
    return jitterize(k_unary, lhp, no_jitter)

import scipy.spatial.distance
def kernel_exponential_ard(X_train, X_test, lhp, no_jitter):
    """
    variance parameter = \sigma_i^2
    cf unit test below for comparison with kernel_exponential_unary
    """
    global dtype
    p = scipy.spatial.distance.cdist(X_train, X_test, metric='mahalanobis', VI=np.diag(1/np.exp(lhp['variances'])))
    k_unary = learn_predict.dtype_for_arrays(np.exp(lhp["unary"]) * np.exp( (-1/2) * np.square(p)))
    return jitterize(k_unary, lhp, no_jitter)

def jitterize(k_unary, lhp, no_jitter):
    if no_jitter:
        return k_unary
    else:
        return k_unary + (np.exp(lhp["jitter"])) * np.eye(k_unary.shape[0])
    
def compute_kernels_from_data(kernel, lhp, X_train, X_test, n_labels):
    k_unary = kernel(X_train, X_train, lhp, no_jitter=False)
#    read_randoms(len(k_unary.flatten(order='F')), should=k_unary.flatten(order='F'), true_random_source=False) # DEBUG
    lower_chol_k_compact = gram_compact(np.linalg.cholesky(k_unary), np.sqrt(np.exp(lhp["binary"])), n_labels)
    # NB no jitter for prediction; because no need to take chol or inv
    '''
    must compute S = K* ' K^-1, equivalent to S K = K*', ie K' S' = K*, ie K S' = K* (cos K sym)
    x=solve(A,b) returns x solution of Ax=b
    so solve(K, K*) gives S', hence need to transpose the result 
    '''
    k_star_unary = kernel(X_train, X_test, lhp, no_jitter=True)
    k_star_T_k_inv = gram_compact(np.linalg.solve(k_unary, k_star_unary).T, gram_binary_scalar=1, n_labels=n_labels)
    
    #read_randoms(should=k_star_T_k_inv_unary.ravel(order='F'), true_random_source=False) #DEBUG

    #===========================================================================
    #if (n_f_star > 0):
    # % cholcov( k** - k*' K^-1 k* )
    # lowerCholfStarCov = chol(exp(lhp(1)) * (X_test * X_test') ...
    #     + noise_param * eye(TT_test) ... % jitter not needed in theory, but in practice needed for numerical stability of chol() operation
    #     - k_star_T_k_inv_unary * kStar_unaryT')';
    #===========================================================================
    #del k_star_unary # not needed since we're in local scope anyway

    return (lower_chol_k_compact, k_star_T_k_inv)
    
class gram_compact():
    def __init__(self, gram_unary, gram_binary_scalar, n_labels):
        """
        gram_unary could be eg kStarTKInv_unary
        gram_binary_scalar eg just binary hyperparameter; binary part of matrix assumed = gram_binary_scalar * eye(n_labels **2)
        """
        self.gram_unary = gram_unary
        self.gram_binary_scalar = gram_binary_scalar
        self.n_labels = n_labels
        self.n_star = gram_unary.shape[0]
        self.n = gram_unary.shape[1]
        
    def expand(self):
        """
            returns an expanded version of this object
        """
        l = [self.gram_unary]*self.n_labels # [k_unary, k_unary, ... k_unary], with length n_label
        l.append(self.gram_binary_scalar * np.eye(n_labels **2))
        return learn_predict.dtype_for_arrays(scipy.linalg.block_diag(*tuple(l)))
    
    def cholesky_T(self):
        try:
            return gram_compact(np.linalg.cholesky(self.gram_unary).T, np.sqrt(self.gram_binary_scalar), self.n_labels)
        except np.linalg.LinAlgError as  e:
            import matplotlib.pyplot as plt
            plt.matshow(self.gram_unary)
            print(self.gram_unary)
            raise e
            
    
    def T_solve(self, v):
        """
        equivalent forms of x = K.T_solve(v):
        x = numpy.linalg.solve(K.T, v)
        K.T * x = v
        Matlab x = K'\v
        """
        assert(v.shape[0] == (self.n_labels * self.n + self.n_labels ** 2))
        result = np.zeros((self.n_labels * self.n + self.n_labels ** 2), dtype=learn_predict.dtype_for_arrays)
        for label in range(self.n_labels):
            result[label * self.n : (label + 1) * self.n] = np.linalg.solve(self.gram_unary.T, v[label*self.n : (label+1)*self.n])
        result[self.n*self.n_labels:] = v[self.n*self.n_labels:] / self.gram_binary_scalar # binary section, should be length n_labels ** 2
        return result
    
    def solve(self, v):
        """
        equivalent forms of x = K.solve(v):
        x = numpy.linalg.solve(K, v)
        K * x = v
        Matlab x = K\v
        """
        assert(v.shape[0] == (self.n_labels * self.n + self.n_labels ** 2))
        result = np.zeros((self.n_labels * self.n + self.n_labels ** 2), dtype=learn_predict.dtype_for_arrays)
        for label in range(self.n_labels):
            result[label * self.n : (label + 1) * self.n] = np.linalg.solve(self.gram_unary, v[label*self.n : (label+1)*self.n])
        result[self.n*self.n_labels:] = v[self.n*self.n_labels:] / self.gram_binary_scalar # binary section, should be length n_labels ** 2
        return result
    
    def dot_wrapper(self, v, A):
        assert(v.shape[0] == (self.n_labels * self.n + self.n_labels ** 2))
        result = np.zeros((self.n_labels * self.n_star + self.n_labels ** 2), dtype=learn_predict.dtype_for_arrays)
        for label in range(self.n_labels):
            result[label * self.n_star : (label + 1) * self.n_star] = np.dot(A, v[label*self.n : (label+1)*self.n])
            # maybe can rewrite this by properly shaping the values in f, and then doing a single dot()
            # but this is no performance bottleneck so leave it
        result[self.n_star*self.n_labels:] = v[self.n*self.n_labels:] * self.gram_binary_scalar # binary section, should be length n_labels ** 2
        return result
        
    def dot(self, v):
        return self.dot_wrapper(v, self.gram_unary)
        
    def T_dot(self, v):
        return self.dot_wrapper(v, self.gram_unary.T)
        
    def diag_log_sum(self):
        """
            return sum(log(diag(self)))
        """
        return np.log(np.diag(self.gram_unary)).sum() * self.n_labels + np.log(self.gram_binary_scalar) * self.n_labels ** 2
        
    def inv_add_diag_cholesky_T(self, scalar):
        """
        K.inv_add_diag_cholesky_T(s) = Matlab lower_chol(K^-1 + diag(s))
        """
        assert(self.n == self.n_star)
        K_inv = np.linalg.inv(self.gram_unary)
        diag_matrix = np.eye(self.n) * scalar
        return gram_compact(np.linalg.cholesky(K_inv + diag_matrix).T, np.sqrt(1 / self.gram_binary_scalar + scalar), self.n_labels)



if __name__ == "__main__":
    import numpy
    import scipy.linalg

    numpy.random.seed(0)
    a = numpy.random.rand(10,10)
    u = numpy.linalg.cholesky(a.T.dot(a)).T
    v = numpy.random.rand(10)
    numpy.testing.assert_array_almost_equal(
        numpy.linalg.solve(u,v),
        scipy.linalg.solve_triangular(u,v))
    
    # test gram_compact
    n_labels = 3
    n = 10
    n_star = 11
    k_unary = learn_predict.dtype_for_arrays(np.random.rand(n_star, n)) # like kStarTKInv_unary
    k_binary_scalar = learn_predict.dtype_for_arrays(np.random.rand()**2) # equivalent to lhp['binary']
    v = learn_predict.dtype_for_arrays(np.random.rand(n * n_labels + n_labels **2))
    k_compact = gram_compact(k_unary, k_binary_scalar, n_labels)
    
    np.testing.assert_almost_equal(
        k_compact.expand().dot(v),
        gram_compact(k_unary, k_binary_scalar, n_labels).dot(v),
        decimal=5)
        
    k_unary = learn_predict.dtype_for_arrays(np.random.rand(n, n)) # now square
    k_unary = k_unary.T.dot(k_unary) # make it pos def
    k_compact = gram_compact(k_unary, k_binary_scalar, n_labels)
    np.testing.assert_almost_equal(
        np.linalg.solve(k_compact.expand().T, v),
        gram_compact(k_unary, k_binary_scalar, n_labels).T_solve(v),
        decimal=5)
    np.testing.assert_almost_equal(
        np.linalg.solve(k_compact.expand(), v),
        gram_compact(k_unary, k_binary_scalar, n_labels).solve(v),
        decimal=5)
    np.testing.assert_almost_equal(
        np.linalg.cholesky(k_compact.expand()).T,
        gram_compact(k_unary, k_binary_scalar, n_labels).cholesky_T().expand(),
        decimal=5)
    s = 7
    np.testing.assert_almost_equal(
        np.linalg.cholesky(np.linalg.inv(k_compact.expand()) 
            + np.eye(n_labels * n + n_labels ** 2) * s).T,
        gram_compact(k_unary, k_binary_scalar, n_labels).inv_add_diag_cholesky_T(s).expand(),
        decimal=5)
    np.testing.assert_almost_equal(
        k_compact.expand().T.dot(v),
        gram_compact(k_unary, k_binary_scalar, n_labels).T_dot(v),
        decimal=5) # test works because this k_unary is not symmetric
    np.testing.assert_almost_equal(
        np.sum(np.log(np.diag(k_compact.expand()))),
        gram_compact(k_unary, k_binary_scalar, n_labels).diag_log_sum(),
        decimal=4)

    # test ARD exponential kernel
    X_train = np.array([[0]])
    X_test = np.array([[3]])
    np.testing.assert_approx_equal(kernel_exponential_ard(X_train, X_test, {'unary' : np.log(1), 'variances' : np.log([2])}, no_jitter=True),
                        np.exp(-1/2 * (1/2) * 3**2))
    X_train = np.array([[0,2]])
    X_test = np.array([[3,-5]])
    np.testing.assert_approx_equal(kernel_exponential_ard(X_train, X_test, {'unary' : np.log(6), 'variances' : np.log([2, 5])}, no_jitter=True),
                        6 * np.exp(-1/2 * ((1/2) * 3**2 + (1/5) * 7**2)))
    np.testing.assert_approx_equal(
        kernel_exponential_unary(X_train, X_test, {'unary': np.log(1), 'length_scale': np.log(7)}, no_jitter=True),
        kernel_exponential_ard(X_train, X_test, {'unary': np.log(1), 'variances' : np.log([7**2, 7**2])}, no_jitter=True)
        )
        