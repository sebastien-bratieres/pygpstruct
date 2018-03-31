import numpy as np
import scipy
import util

import scipy.sparse.csr
def kernel_linear(X_train, X_test, lhp, no_jitter):
    p = np.dot(X_train,X_test.T)
    if isinstance(p, scipy.sparse.csr.csr_matrix):
        p = p.toarray() # cos if using X_train sparse vector, p will be a csr_matrix -- incidentally in this case the resulting k_unary cannot be flattened, it will result in a (1,X) 2D matrix !
    k_unary = np.array(p, dtype=util.dtype_for_arrays)
    return jitterize(k_unary, lhp, no_jitter)
    
def kernel_linear_ard(X_train, X_test, lhp, no_jitter):
    if isinstance(X_train,scipy.sparse.coo_matrix) :
        X_train_premult = X_train.multiply(scipy.sparse.coo_matrix(np.exp(lhp['variances']))) # shapes: (n_data_train, n_features) * (n_features) which will be broadcast -> (n_data_train, n_features)
        p = np.dot(X_train_premult,X_test.T)
        p = p.toarray() # cos if using X_train sparse vector, p will be a csr_matrix -- incidentally in this case the resulting k_unary cannot be flattened, it will result in a (1,X) 2D matrix !
    else:
        X_train_premult = np.multiply(X_train, np.exp(lhp['variances'])) # shapes: (n_data_train, n_features) * (n_features) which will be broadcast -> (n_data_train, n_features)
        p = np.dot(X_train_premult,X_test.T)        
    k_unary = np.array(p, dtype=util.dtype_for_arrays)
    return jitterize(k_unary, lhp, no_jitter)

import sklearn.metrics.pairwise
def kernel_exponential(X_train, X_test, lhp, no_jitter):
    """
    $k_\text{exponential}(\mathbf{x},\mathbf{x'}) = \exp({-\frac{1}{2 \ell ^2} \Vert \mathbf{x} - \mathbf{x'} \Vert ^2})$  
    parameter 'length_scale' is log \ell
    """
    p = sklearn.metrics.pairwise.euclidean_distances(X_train, X_test, squared=True)
    # sames as scipy.spatial.distance.cdist(X_train,X_test, 'sqeuclidean')
    # but works with Scipy sparse and Numpy dense arrays
    # I thought it would be equal to X_train.dot(X_train.T) + X_test.dot(X_test.T) - X_train.dot(X_test.T) - X_test.dot(X_train.T))
    # but it doesnt seem to
    k_unary = util.dtype_for_arrays(np.exp( -(1/2) * 1/(np.exp(lhp["length_scale"])**2 ) * p))
    return jitterize(k_unary, lhp, no_jitter)


import numba
@numba.jit
def ard_outer_loop(n_data_train, n_data_test, variances, cache):
    p = np.empty((n_data_train, n_data_test), dtype=util.dtype_for_arrays) # 2% faster to allocate here, instead of pre-allocating in cache run (and using kernel_exponential_ard.p)
    for i_train in range(n_data_train):
        for i_test in range(n_data_test):
            p[i_train, i_test] = np.sum(ard_inner_func(cache.row_train_cache[i_train], cache.row_test_cache[i_test], variances))
    
    return util.dtype_for_arrays(np.exp(p)) # 2% faster not to use in-place exp

@numba.vectorize(['float64(float64, float64, float64)', 'float32(float32, float32, float32)'])
def ard_inner_func(a, b, v):
    return (-1/2) * (a-b)**2/v


import scipy.spatial.distance
def kernel_exponential_ard(X_train, X_test, lhp, no_jitter):
    """
    $k_\text{exponential ARD}(\mathbf{x},\mathbf{x'}) = \exp({-\frac{1}{2} \sum_i \frac{1}{\sigma_i ^2} ( \mathbf{x}_i - \mathbf{x'}_i ) ^2})$  
    parameter 'variance' is log(\sigma_i^2)_i
    cf unit test below for comparison with kernel_exponential_unary
    will attempt to convert sparse X_train, X_test to np.array, risking a memory error: use with caution with large arrays!
    sparse matrix optimization works on assumption that X_train, X_test don't change between calls
    """
    #print("kernel evaluation, lhp=%s" % str(lhp))
    variances = np.exp(lhp['variances'])
    if isinstance(X_train,scipy.sparse.coo_matrix) :
        n_data_train = X_train.shape[0]
        n_data_test = X_test.shape[0]
        
        key = str(n_data_train) + str(n_data_test)
        if key in kernel_exponential_ard.dict:
            cache = kernel_exponential_ard.dict[key]
        else:
            class cache:
                pass
            kernel_exponential_ard.dict[key] = cache()
            cache.row_test_cache = [None] * n_data_test # initialize empty list to memoize results of repeated X_test.getrow
            for i_test in range(n_data_test):
                cache.row_test_cache[i_test] = X_test.getrow(i_test).toarray()  # stores result for later lookup 
                # TODO works only for segmentation, where it is feasible to do .toarray on feature vector!
            cache.row_train_cache = [None] * n_data_train
            for i_train in range(n_data_train):
                cache.row_train_cache[i_train] = X_train.getrow(i_train).toarray()
        k_unary = ard_outer_loop(n_data_train, n_data_test, variances, cache)
    else:
        p = scipy.spatial.distance.cdist(X_train, X_test, metric='mahalanobis', VI=np.diag(1/variances))
        k_unary = util.dtype_for_arrays(np.exp( (-1/2) * np.square(p)))
    return jitterize(k_unary, lhp, no_jitter)
kernel_exponential_ard.dict = {}

def jitterize(k_unary, lhp, no_jitter):
    if no_jitter:
        return k_unary
    else:
        return k_unary + (np.exp(lhp["jitter"])) * np.eye(k_unary.shape[0], dtype=util.dtype_for_arrays)
    
def compute_lower_chol_k(kernel, lhp, X_train, n_labels):
    k_unary = kernel(X_train, X_train, lhp, no_jitter=False)
#    read_randoms(len(k_unary.flatten(order='F')), should=k_unary.flatten(order='F'), true_random_source=False) # DEBUG
    if ("unary" in lhp and "binary" in lhp):
        lower_chol_k_compact = gram_compact(np.linalg.cholesky(np.exp(lhp["unary"]) * k_unary), np.sqrt(np.exp(lhp["binary"])), n_labels)
    elif ("alpha" in lhp and "lambda" in lhp):
        lower_chol_k_compact = gram_compact(np.linalg.cholesky(np.exp(lhp["alpha"]) * k_unary), np.exp(0.5 * (lhp["alpha"] + lhp["lambda"])), n_labels)
        # K = alpha ( k_x + lambda k_y) = exp(log alpha) + exp(log alpha+ log lambda) k_y
        # so K is block-wise: exp(log alpha) * gram_unary, exp(log alpha+ log lambda) * gram_binary, with gram_binary=I
        # so chol_K consists of block-wise: chol(exp(log alpha) * gram_unary), sqrt(exp(log alpha+ log lambda)) * chol(I) = exp(0.5 * (...)) * I
    else:
        raise NameError("Unknown parameterization for kernel")
    return lower_chol_k_compact

def compute_k_star_T_k_inv(kernel, lhp, X_train, X_test, n_labels, lower_chol_k_compact):
    # NB no jitter for prediction; because no need to take chol or inv
    '''
    must compute S = K* ' K^-1, equivalent to S K = K*', ie K' S' = K*, ie K S' = K* (cos K sym)
    x=solve(A,b) returns x solution of Ax=b
    so solve(K, K*) gives S', hence need to transpose the result 
    '''
    k_star_unary = kernel(X_train, X_test, lhp, no_jitter=True)
    k_unary = lower_chol_k_compact.gram_unary.dot(lower_chol_k_compact.gram_unary.T)
    k_star_T_k_inv = gram_compact(np.linalg.solve(k_unary, k_star_unary).T, gram_binary_scalar=1, n_labels=n_labels)
    # cancel out: the multiplicative factors hp_unary, hp_binary (not even considered in previous line)
    
    #read_randoms(should=k_star_T_k_inv_unary.ravel(order='F'), true_random_source=False) #DEBUG

    #===========================================================================
    #if (n_f_star > 0):
    # % cholcov( k** - k*' K^-1 k* )
    # lowerCholfStarCov = chol(exp(lhp(1)) * (X_test * X_test') ...
    #     + noise_param * eye(TT_test) ... % jitter not needed in theory, but in practice needed for numerical stability of chol() operation
    #     - k_star_T_k_inv_unary * kStar_unaryT')';
    #===========================================================================
    #del k_star_unary # not needed since we're in local scope anyway

    return k_star_T_k_inv

def compute_kernels_from_data_AFAC(kernel, lhp, X_train, X_test, n_labels):
    k_unary = kernel(X_train, X_train, lhp, no_jitter=False)
#    read_randoms(len(k_unary.flatten(order='F')), should=k_unary.flatten(order='F'), true_random_source=False) # DEBUG
    if (lhp.has_key["unary"] and lhp.has_key["binary"]):
        lower_chol_k_compact = gram_compact(np.linalg.cholesky(np.exp(lhp["unary"]) * k_unary), np.sqrt(np.exp(lhp["binary"])), n_labels)
    elif (lhp.has_key["alpha"] and lhp.has_key["lambda"]):
        lower_chol_k_compact = gram_compact(np.linalg.cholesky(np.exp(lhp["alpha"]) * k_unary), np.sqrt(np.exp(lhp["alpha"]) + lhp["gamma"]), n_labels)
    else:
        raise("Unknown parameterization for kernel")
    
    # NB no jitter for prediction; because no need to take chol or inv
    '''
    must compute S = K* ' K^-1, equivalent to S K = K*', ie K' S' = K*, ie K S' = K* (cos K sym)
    x=solve(A,b) returns x solution of Ax=b
    so solve(K, K*) gives S', hence need to transpose the result 
    '''
    k_star_unary = kernel(X_train, X_test, lhp, no_jitter=True)
    k_star_T_k_inv = gram_compact(np.linalg.solve(k_unary, k_star_unary).T, gram_binary_scalar=1, n_labels=n_labels)
    # cancel out: the multiplicative factors hp_unary, hp_binary (not even considered in previous line)
    
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
        l.append(self.gram_binary_scalar * np.eye(self.n_labels **2))
        return util.dtype_for_arrays(scipy.linalg.block_diag(*tuple(l)))
    
    def T(self):
        return gram_compact(self.gram_unary.T, self.gram_binary_scalar, self.n_labels)
        
    def cholesky_T(self):
        """
        upper Cholesky
        """
        try:
            return gram_compact(np.linalg.cholesky(self.gram_unary).T, np.sqrt(self.gram_binary_scalar), self.n_labels)
        except np.linalg.LinAlgError as  e: # matrix is probably singular
            import matplotlib.pyplot as plt
            plt.matshow(self.gram_unary)
            print(self.gram_unary)
            raise e

    def cholesky(self):
        """
        lower Cholesky
        """
        try:
            return gram_compact(np.linalg.cholesky(self.gram_unary), np.sqrt(self.gram_binary_scalar), self.n_labels)
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
        result = np.zeros((self.n_labels * self.n + self.n_labels ** 2), dtype=util.dtype_for_arrays)
        for label in range(self.n_labels):
            result[label * self.n : (label + 1) * self.n] = np.linalg.solve(self.gram_unary.T, v[label*self.n : (label+1)*self.n])
        result[self.n*self.n_labels:] = v[self.n*self.n_labels:] / self.gram_binary_scalar # binary section, should be length n_labels ** 2
        return result
    
    def solve(self, v):
        """
        return self^-1 * v
        equivalent forms of x = K.solve(v):
        x = numpy.linalg.solve(K, v)
        K * x = v
        Matlab x = K\v
        """
        assert(v.shape[0] == (self.n_labels * self.n + self.n_labels ** 2))
        result = np.zeros((self.n_labels * self.n + self.n_labels ** 2), dtype=util.dtype_for_arrays)
        for label in range(self.n_labels):
            result[label * self.n : (label + 1) * self.n] = np.linalg.solve(self.gram_unary, v[label*self.n : (label+1)*self.n])
        result[self.n*self.n_labels:] = v[self.n*self.n_labels:] / self.gram_binary_scalar # binary section, should be length n_labels ** 2
        return result
    
    def solve_triangular(self, v):
        """
        return self^-1 * v, when we know that self is lower triangular
        equivalent forms of x = K.solve(v):
        x = numpy.linalg.solve(K, v)
        K * x = v
        Matlab x = K\v
        """
        assert(v.shape[0] == (self.n_labels * self.n + self.n_labels ** 2))
        result = np.zeros((self.n_labels * self.n + self.n_labels ** 2), dtype=util.dtype_for_arrays)
        for label in range(self.n_labels):
            result[label * self.n : (label + 1) * self.n] = scipy.linalg.solve_triangular(self.gram_unary, v[label*self.n : (label+1)*self.n], lower=True, check_finite=False)
        result[self.n*self.n_labels:] = v[self.n*self.n_labels:] / self.gram_binary_scalar # binary section, should be length n_labels ** 2
        return result

    @staticmethod
    def solve_cholesky_lower_basic(L, v):
        return scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L,v, lower=True, check_finite=False), lower=False, check_finite=False)
        # when you know your matrix is triangular, don't use (scipy or numpy).linalg.solve, but rather scipy.linalg.solve_triangular, with the check_finite parameter.
        # same answer given by: return np.linalg.solve(L.T, np.linalg.solve(L,v)) 
        
    def solve_cholesky_lower(self, v):
        """
        Solve A*X = B for X, where A is square, symmetric, positive definite. The input to the function is L, the lower Cholesky decomposition of A (ie A = L L^T), and the matrix B.
        Therefore, for self=L, this function returns (L L^T)^-1 B = L^-T L^-1 B
        "self" is assumed to be the lower triangular L
        v must be either of type gram_compact, or a vector.
        
        L = A.cholesky()
        X = L.solve_cholesky_lower(v) == A.solve(v) == L.T_solve(L.solve(v))
        """
        np.testing.assert_array_almost_equal(self.gram_unary, np.tril(self.gram_unary)) # assert gram_unary is lower triangular
        if type(v) is gram_compact:
            return gram_compact(gram_compact.solve_cholesky_lower_basic(self.gram_unary, v.gram_unary),
                                v.gram_binary_scalar / (self.gram_binary_scalar ** 2),
                                self.n_labels) # result is no longer lower triangular like self
        else:
            assert(v.shape[0] == (self.n_labels * self.n + self.n_labels ** 2))
            result = np.zeros((self.n_labels * self.n + self.n_labels ** 2), dtype=util.dtype_for_arrays)
            for label in range(self.n_labels):
                result[label * self.n : (label + 1) * self.n] = gram_compact.solve_cholesky_lower_basic(self.gram_unary, v[label*self.n : (label+1)*self.n])
            result[self.n*self.n_labels:] = v[self.n*self.n_labels:] / (self.gram_binary_scalar ** 2) # binary section, should be length n_labels ** 2
        return result
# REFACTOR gram_compact: make subclass lower_tri holding lower triang unary (typically as a result of a .cholesky() operation)
# then for this class, can use solve_triang instead of solve_ and define solve_chol_lower
        
    def dot_wrapper(self, v, A):
        assert(v.shape[0] == (self.n_labels * self.n + self.n_labels ** 2))
        result = np.zeros((self.n_labels * self.n_star + self.n_labels ** 2), dtype=util.dtype_for_arrays)
        for label in range(self.n_labels):
            result[label * self.n_star : (label + 1) * self.n_star] = np.dot(A, v[label*self.n : (label+1)*self.n])
            # maybe can rewrite this by properly shaping the values in f, and then doing a single dot()
            # but this is no performance bottleneck so leave it
        result[self.n_star*self.n_labels:] = v[self.n*self.n_labels:] * self.gram_binary_scalar # binary section, should be length n_labels ** 2
        return result
        
    def dot(self, v):
        if type(v) is gram_compact:
            return gram_compact(self.gram_unary.dot(v.gram_unary),
                                self.gram_binary_scalar * v.gram_binary_scalar,
                                self.n_labels)
        else:
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
        K.inv_add_diag_cholesky_T(s) = Matlab upper chol(K^-1 + diag(s))
        """
        assert(self.n == self.n_star)
        K_inv = np.linalg.inv(self.gram_unary)
        diag_matrix = np.eye(self.n) * scalar
        return gram_compact(np.linalg.cholesky(K_inv + diag_matrix).T, np.sqrt(1 / self.gram_binary_scalar + scalar), self.n_labels)
    
    def __add__(self,right):
        if type(right) is not gram_compact:
            raise TypeError('unsupported operand type(s) for +'+
                            ': \''+type_as_str(self)+'\' and \''+type_as_str(right)+'\'')        
        assert(self.gram_unary.shape[0] == right.gram_unary.shape[0] and self.n_labels == right.n_labels), \
               'gram_compact.__add__: left operand has different dimensions than right operand'
        return gram_compact(self.gram_unary + right.gram_unary, self.gram_binary_scalar + right.gram_binary_scalar, self.n_labels)

    def __sub__(self, right):
        if type(right) is not gram_compact:
            raise TypeError('unsupported operand type(s) for -'+
                            ': \''+type_as_str(self)+'\' and \''+type_as_str(right)+'\'')        
        assert(self.gram_unary.shape[0] == right.gram_unary.shape[0] and self.n_labels == right.n_labels), \
               'gram_compact.__sub__: left operand has different dimensions than right operand'
        return gram_compact(self.gram_unary - right.gram_unary, self.gram_binary_scalar - right.gram_binary_scalar, self.n_labels)

    @staticmethod
    def identity(n_train, n_labels, scale):
        return gram_compact(scale * np.eye(n_train, n_train), scale, n_labels) 
        
if __name__ == "__main__":
    import numpy
    import scipy.linalg

    # test kernel_exponential_ard acceleration for sparse matrices
    n_data_train=10
    n_data_test=10
    n_features=100
    lhp = {'variances' : np.log(np.random.rand(n_features)), 'jitter' : np.log(1e-4)}

    import scipy.sparse
    x_train=scipy.sparse.coo_matrix(np.random.binomial(n=1, p=0.05, size=(n_data_train, n_features)))
    x_test=scipy.sparse.coo_matrix(np.random.binomial(n=1, p=0.05, size=(n_data_test, n_features)))

    import numpy.testing
    # kernel_exponential gives same result as kernel_exponential_ard with variances = log ones()
    lhp['length_scale'] = np.log(1)
    kernel_exponential_ard.dict = {}
    np.testing.assert_array_almost_equal(
        kernel_exponential_ard(x_train.toarray(), x_test.toarray(), {'variances': np.log(np.ones(n_features))}, no_jitter=True),
        kernel_exponential(x_train, x_test, lhp, no_jitter=True)
        )
    # result of kernel_exponential_ard is the same for sparse and non-sparse
    kernel_exponential_ard.dict = {}
    np.testing.assert_array_almost_equal(
        kernel_exponential_ard(x_train, x_test, lhp, no_jitter=True),
        kernel_exponential_ard(x_train.toarray(), x_test.toarray(), lhp, no_jitter=True)
        )    
    
    # result of kernel_linear_ard is the same for sparse and non-sparse
    np.testing.assert_array_almost_equal(
        kernel_linear_ard(x_train, x_test, lhp, no_jitter=True),
        kernel_linear_ard(x_train.toarray(), x_test.toarray(), lhp, no_jitter=True)
        )
    # kernel_linear gives same result as kernel_linear_ard with variances = ones()
    np.testing.assert_array_almost_equal(
        kernel_linear_ard(x_train, x_test, {'variances': np.log(np.ones(n_features))}, no_jitter=True),
        kernel_linear(x_train.toarray(), x_test.toarray(), lhp, no_jitter=True)
        )

    # other tests

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
    k_unary = util.dtype_for_arrays(np.random.rand(n_star, n)) # like kStarTKInv_unary
    k_binary_scalar = util.dtype_for_arrays(np.random.rand()**2) # equivalent to lhp['binary']
    v = util.dtype_for_arrays(np.random.rand(n * n_labels + n_labels **2))
    k_compact = gram_compact(k_unary, k_binary_scalar, n_labels)
    
    # test constructor, .expand()
    np.testing.assert_almost_equal(
        k_compact.expand().dot(v),
        gram_compact(k_unary, k_binary_scalar, n_labels).dot(v),
        decimal=5)
        
    k_unary = util.dtype_for_arrays(np.random.rand(n, n)) # now square
    k_unary = k_unary.T.dot(k_unary) # make it pos def
    k_compact = gram_compact(k_unary, k_binary_scalar, n_labels)
    # test .T_solve
    np.testing.assert_almost_equal(
        np.linalg.solve(k_compact.expand().T, v),
        gram_compact(k_unary, k_binary_scalar, n_labels).T_solve(v),
        decimal=5)
    # test .solve
    np.testing.assert_almost_equal(
        np.linalg.solve(k_compact.expand(), v),
        gram_compact(k_unary, k_binary_scalar, n_labels).solve(v),
        decimal=5)
    k_unary_lower = np.tril(util.dtype_for_arrays(np.random.rand(n, n))) # now lower triangular
    L = gram_compact(k_unary_lower, np.sqrt(k_binary_scalar), n_labels)
    A = gram_compact(k_unary_lower.dot(k_unary_lower.T), k_binary_scalar, n_labels)
    
    # test .solve_triangular
    # .solve and .solve_triangular should give equal results
    np.testing.assert_almost_equal(
        L.solve_triangular(v),
        np.linalg.solve(L.expand(), v),
        decimal=5)
    
    # test .cholesky
    np.testing.assert_almost_equal(
        L.expand(),
        A.cholesky().expand(),
        decimal=5)
    
    # check L * L.T == A
    np.testing.assert_almost_equal(
        L.expand().dot(L.expand().T),
        A.expand(),
        decimal=5)
    
    # test .solve_chol_lower_basic
    # A * solve_chol(L, C) == A * A^-1 C == C
    C = np.random.rand(n * n_labels + n_labels **2, n * n_labels + n_labels **2)
    np.testing.assert_almost_equal(
        A.expand().dot(gram_compact.solve_cholesky_lower_basic(L.expand(), C)),
        C,
        decimal=4)

    # test .solve_chol_lower(vector)
    # solve_chol(L, v) == A^-1 v == L^-T L^-1 v
    np.testing.assert_almost_equal(
        L.solve_cholesky_lower(v),
        L.T_solve(L.solve(v)),
        decimal=4)

    # test .solve_chol_lower(matrix)
    # solve_chol(L, A) == I
    np.testing.assert_almost_equal(
        L.solve_cholesky_lower(A).expand(),
        gram_compact.identity(n, n_labels, 1.0).expand(),
        decimal=4)
    np.testing.assert_almost_equal(
        A.solve(v),
        np.linalg.solve(A.expand(), v),
        decimal=5)

    np.testing.assert_almost_equal(
        np.linalg.solve(L.expand().T, np.linalg.solve(L.expand(), v)),
        np.linalg.solve(A.expand(), v),
        decimal=3)
    
    # test .solve_chol_lower
    np.testing.assert_almost_equal(
        L.solve_cholesky_lower(v),
        A.solve(v),
        decimal=3)
    # test .cholesky_T
    np.testing.assert_almost_equal(
        np.linalg.cholesky(k_compact.expand()).T,
        gram_compact(k_unary, k_binary_scalar, n_labels).cholesky_T().expand(),
        decimal=5)

    # test .inv_add_diag_cholesky_T
    s = 7
    np.testing.assert_almost_equal(
        np.linalg.cholesky(np.linalg.inv(k_compact.expand()) 
            + np.eye(n_labels * n + n_labels ** 2) * s).T,
        gram_compact(k_unary, k_binary_scalar, n_labels).inv_add_diag_cholesky_T(s).expand(),
        decimal=5)
    # test .T_dot
    np.testing.assert_almost_equal(
        k_compact.expand().T.dot(v),
        gram_compact(k_unary, k_binary_scalar, n_labels).T_dot(v),
        decimal=5) # test works because this k_unary is not symmetric
    # test .diag_log_sum
    np.testing.assert_almost_equal(
        np.sum(np.log(np.diag(k_compact.expand()))),
        gram_compact(k_unary, k_binary_scalar, n_labels).diag_log_sum(),
        decimal=4)
    
    # test .dot(vector)
    np.testing.assert_almost_equal(
        A.dot(v),
        A.expand().dot(v), # using np.ndarray.dot
        decimal=6)
    
    # test .dot(matrix)
    np.testing.assert_almost_equal(
        A.dot(A).expand(),
        A.expand().dot(A.expand()), # using np.ndarray.dot
        decimal=6)

    #test .__add__
    np.testing.assert_almost_equal(
        (A + L).expand(),
        A.expand() + L.expand(), # using np.ndarray.__add__
        decimal=7)
        
    # test .__sub__
    np.testing.assert_almost_equal(
        (A - L).expand(),
        A.expand() - L.expand(), # using np.ndarray.__sub__
        decimal=7)

    # test .identity
    np.testing.assert_almost_equal(
        gram_compact.identity(n_train = 10, n_labels = 5, scale = 7).expand(),
        7.0 * np.eye(10 * 5 + 5 ** 2),
        decimal = 8)
        
    # test ARD exponential kernel
    X_train = np.array([[0]])
    X_test = np.array([[3]])
    np.testing.assert_approx_equal(kernel_exponential_ard(X_train, X_test, {'unary' : np.log(1), 'variances' : np.log([2])}, no_jitter=True),
                        np.exp(-1/2 * (1/2) * 3**2))
    X_train = np.array([[0,2]])
    X_test = np.array([[3,-5]])
    np.testing.assert_approx_equal(kernel_exponential_ard(X_train, X_test, {'unary' : np.log(1), 'variances' : np.log([2, 5])}, no_jitter=True),
                        np.exp(-1/2 * ((1/2) * 3**2 + (1/5) * 7**2)))
    np.testing.assert_approx_equal(
        kernel_exponential(X_train, X_test, {'unary': np.log(1), 'length_scale': np.log(7)}, no_jitter=True),
        kernel_exponential_ard(X_train, X_test, {'unary': np.log(1), 'variances' : np.log([7**2, 7**2])}, no_jitter=True)
        )
        
"""
# could be useful once

# kernel computation using linear kernel (dot product) for sparse COO matrices

n_data=10
n_data_test=3
n_features=100
import scipy.sparse
x_train=scipy.sparse.coo_matrix(np.random.randint(low=0, high=3, size=(n_data, n_features)))
x_test=scipy.sparse.coo_matrix(np.random.randint(low=0, high=3, size=(n_data_test, n_features)))
print(x_train.shape)
#print(x_train)
k = np.zeros((n_data, n_data_test))
for coord in range(n_features):
    v_train = x_train.T.getrow(coord)
    v_test = x_test.T.getrow(coord)
    for (i_data_train, i_train) in enumerate(v_train.indices):
        for (i_data_test, i_test) in enumerate(v_test.indices):
            k[i_train, i_test] += v_train.data[i_data_train] * v_test.data[i_data_test]

print(k)
print(np.dot(x_train.toarray(), x_test.T.toarray()))
import sklearn.metrics.pairwise
print(sklearn.metrics.pairwise.euclidean_distances(x_train, x_test, squared=True))
"""