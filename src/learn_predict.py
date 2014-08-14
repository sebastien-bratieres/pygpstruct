from __future__ import division
from __future__ import print_function
import numpy as np
import logging
import time
import ess_k_sampler
import sys
import os
import numpy.testing
import pickle
import glob

import sklearn.metrics.pairwise
def kernel_exponential_unary(X_train, X_test, lhp, noise_param):
    p = sklearn.metrics.pairwise.euclidean_distances(X_train, X_test, squared=True)
    # sames as scipy.spatial.distance.cdist(X_train,X_test, 'sqeuclidean')
    # but works with Scipy sparse and Numpy dense arrays
    # I thought it would be equal to X_train.dot(X_train.T) + X_test.dot(X_test.T) - X_train.dot(X_test.T) - X_test.dot(X_train.T))
    # but it doesnt seem to
    k_unary = learn_predict_gpstruct.dtype(np.exp(lhp["unary"]) * np.exp( -1/np.exp(lhp["length_scale"]) * p))
    if (noise_param == 0):
        return k_unary
    else:
        return k_unary + (noise_param) * np.eye(k_unary.shape[0])

def read_randoms(n=-1, type=None, should=None, true_random_source=True):
    """
    to intialize this function:
    read_randoms.offset=0 #DEBUG
    read_randoms.file = np.loadtxt('/tmp/sb358/ess_randoms.txt') #DEBUG
    """
    if true_random_source:
        if type != None:
            if type=='u':
                result=read_randoms.prng.rand(n)
            else:
                result=read_randoms.prng.randn(n)
        else:
            return # type==None, but we're generating random numbers so can't check anything
    else:
        if (n == -1 and should != None):
            n = len(should)
        result = read_randoms.file[read_randoms.offset:read_randoms.offset+n]
        if should != None:
            #print("testing start offset : " + str(read_randoms.offset) + ", length : " + str(n))
            try:
                numpy.testing.assert_almost_equal(should, result)
            except AssertionError as e:
                raise e
            
        read_randoms.offset = read_randoms.offset+n
    return learn_predict_gpstruct.dtype(result)

    
# ================================================= 
def learn_predict_gpstruct( prepare_from_data,
                            result_prefix=None, 
                            console_log=True,
                            n_samples=0, 
                            prediction_thinning=1, 
                            hp_thinning=100000, 
                            n_f_star=0, 
                            hp_mode=0, prior=1, hp_bin_init=0.01, noise_param=1e-4,
                            kernel=kernel_exponential_unary,
                            random_seed=0,
                            stop_check=None
                            ):
    """
    result_prefix should end with the desired character to allow result_prefix + string constructions:
    end in / for directory: will put files into result_prefix directory
    end in . for file prefix (ie result_prefix = result_dir + '/' + file_prefix: will put files into result_dir, with filenames prefixed with file_prefix
    
    # n_samples: # f samples, ie # of MCMC iterations
    # prediction_thinning: #samples f (MCMC iterations) after which to compute f* and p(y*) - eg 1 or 10 (the f samples which are not used here are thrown away)
    # hp_thinning: #samples of f after which to resample hyperparams
    # n_f_star: # f* samples given f (0 = MAP) - eg 2
    # hp_mode: 0 for no hp sampling, 1 for prior whitening, 2 for surrogate
    %   data/null aux
    % prior: 1 for narrow, 2 for wide uniform prior
    % hp_bin_init: initial value of binary hyperparameter
    """
    function_args = locals() # store just args passed to function, so as to log them later on
    
    if (glob.glob(result_prefix + 'state.pickle') != []): # hotstart
        hotstart=True # no safety net prevents hotstarting with different parameters. Could do: store parameters dict, check at hotstart that it is identical. Hard to do: just have an "hotstart this" feature which works based on a results folder; implies re-obtain things like dataset, kernel matrices.
    else:
        # make results dir
        # mkdir result_prefix in case it doesnt exist, from http://stackoverflow.com/questions/16029871/how-to-run-os-mkdir-with-p-option-in-python second answer updated !
        hotstart=False
        try:
            os.makedirs(os.path.split(result_prefix)[0])
        except OSError as err:
            if err.errno!=17:
                raise
    # prepare logging
    # logging tree blog post http://rhodesmill.org/brandon/2012/logging_tree/
    logger = logging.getLogger('pyGPstruct') # reuse logger if existing
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers = [] # remove any existing handlers
    
    # create formatter
    formatter = logging.Formatter('%(asctime)sZ - %(levelname)s - %(message)s')
    formatter.converter=time.gmtime # will use GMT timezone for output, hence the Z
    
    # console log handler
    if (console_log):
        ch = logging.StreamHandler(stream=sys.stdout)
        #ch.setLevel(logging.DEBUG) # seems not needed by default 
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    fh = logging.FileHandler(result_prefix + "log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # logging all input parameters cf http://stackoverflow.com/questions/2521901/get-a-list-tuple-dict-of-the-arguments-passed-to-a-function -- bottom line use locals() which is a dict of all local vars
    logger.debug('learn_predict_gpstruct started with arguments: ' + str(function_args))

    
#    if errorThinning < prediction_thinning:
#        logger.error("Value for errorThinning is %g, smaller than prediction_thinning, which is %g. This is illegal. Using %g instead.\n" % errorThinning, prediction_thinning, prediction_thinning)

    lhp = {'unary': np.log(1), 'binary': np.log(hp_bin_init), 'length_scale': np.log(8)}
    if (stop_check == None):
        stop_check = lambda : None # equivalent to pass
    learn_predict_gpstruct.dtype=np.float32
    results = []
    read_randoms.offset=0 #DEBUG
    #read_randoms.file = np.loadtxt('/tmp/sb358/ess_randoms.txt') #DEBUG

    (ll_train, 
     posterior_marginals_test, 
     compute_error_nlm, 
     ll_test, 
     average_marginals, 
     write_marginals,
     read_marginals,
     n_labels, 
     X_train, 
     X_test) = prepare_from_data(logger=logger) 
    # posterior_marginals_test is a function f -> posterior marginals
    # compute_error_nlm(marginals) returns error and avg post marginals
    # ll_test(current f) returns LL of test data
    
    TT_train = X_train.shape[0]
    TT_test = X_test.shape[0]
    #read_randoms(len(X_train.todense().flatten(order='F').T), should=np.squeeze(np.array(X_train.todense()).flatten(order='F').T), true_random_source=False) # DEBUG
    #read_randoms(len(X_test.todense().flatten(order='F').T), should=np.squeeze(np.array(X_test.todense()).flatten(order='F').T), true_random_source=False) # DEBUG
    # prepare kernel matrix
    logger.debug("prepare kernel matrices")

    k_unary = kernel(X_train, X_train, lhp, noise_param)
#    read_randoms(len(k_unary.flatten(order='F')), should=k_unary.flatten(order='F'), true_random_source=False) # DEBUG
    
    k_binary = np.exp(lhp["binary"]) * np.eye(n_labels**2)

    lower_chol_k_unary = np.linalg.cholesky(k_unary)
    lower_chol_k_binary = np.linalg.cholesky(k_binary) # simplify, this is eye
#    print('k*')
    k_star_unary = kernel(X_train, X_test, lhp, noise_param=0)
    # NB no noise for prediction
    k_star_T_k_inv_unary = hashable_compute_kStarTKInv_unary(k_unary, k_star_unary)
    
    #read_randoms(should=k_star_T_k_inv_unary.ravel(order='F'), true_random_source=False) #DEBUG

    del k_unary
    del k_binary
    if (n_f_star > 0):
        pass
    #===========================================================================
    # % cholcov( k** - k*' K^-1 k* )
    # lowerCholfStarCov = chol(exp(lhp(1)) * (X_test * X_test') ...
    #     + noise_param * eye(TT_test) ... % jitter not needed in theory, but in practice needed for numerical stability of chol() operation
    #     - k_star_T_k_inv_unary * kStar_unaryT')';
    #===========================================================================
    del k_star_unary
    
    logger.debug("start MCMC chain")

    if hotstart: # restore state from disk
        with open(result_prefix + 'state.pickle', 'rb') as random_state_file:
            saved_state_dict = pickle.load(random_state_file)
        read_randoms.prng = saved_state_dict['prng']
        current_f = saved_state_dict['current_f']
        mcmc_step = saved_state_dict['mcmc_step']
        current_ll = saved_state_dict['current_ll']
        current_error = saved_state_dict['current_error']
        scaled_ll_test = saved_state_dict['scaled_ll_test']
        avg_error = saved_state_dict['avg_error']
        avg_nlm = saved_state_dict['avg_nlm']
        logger.info('hotstart from iteration %g, including stored random state' % mcmc_step)
    else: # initialize state
        current_f = np.zeros(n_labels * TT_train + n_labels**2, dtype=learn_predict_gpstruct.dtype)
        mcmc_step=0
        read_randoms.prng = np.random.RandomState(random_seed)
        # no need to initialize other variables, since they will be computed during prediction, since we are starting from iteration 0 (for which we are sure prediction will happen)

    while not stop_check() and (mcmc_step < n_samples or n_samples == 0):
        current_f, current_ll = ess_k_sampler.ESS(current_f, ll_train, n_labels, lower_chol_k_unary, lower_chol_k_binary, read_randoms) 
        #read_randoms(should=current_f, true_random_source=False)
        #current_ll = read_randoms(1, should=ll_train(current_f), true_random_source=False)
        
        # prediction : compute f*|D and p(y*|f)
        # - compute f*, then marginals p(y*|f*)
        # - save marginals to disk
        # - read in all marginals so far
        # - discard burnin, from remaining marginals compute Bayesian averaged error rate
        if np.mod(mcmc_step, prediction_thinning) == 0:

#            logger.debug("start prediction")
            # compute mean of f*|D - this involves f (expanded) and k_star_T_k_inv_unary (compact), so need to iterate over n_labels
            f_star_mean = np.zeros(TT_test*n_labels + n_labels**2, dtype=learn_predict_gpstruct.dtype)
            for label in range(n_labels):
                f_star_mean[label*TT_test:(label + 1)*TT_test] = np.dot(k_star_T_k_inv_unary, current_f[label*TT_train:(label+1)*TT_train])
            # maybe can rewrite this by properly shaping the values in f, and then doing a single dot()
            # but this is no performance bottleneck so leave it
            f_star_mean[TT_test*n_labels:] = current_f[TT_train*n_labels:] # binaries are not computed just copied
            if n_f_star == 0:
                marginals_f = posterior_marginals_test(f_star_mean)
            # else:
            # % sample f*
            # % want f* sampling to not affect f sampling, so preserve randn state before this
            # f_rng_state = randn('state');
            # marginals = zeros(TT_test, n_labels, n_f_star);
            # for i=1:n_f_star
            #     marginals(:,:,i) = predictiveMarginalsN(f_star_mean + ...
            #         [sampleFFromKernel(n_labels, lowerCholfStarCov, true ) ; ...
            #         zeros(n_labels^2, 1)]);
            #     % must pad output of sample... cos it's applied to unary
            #     % matrices, while f*mean has unary and bin
            #     % NOTE: using the binaries from f, cos they are not
            #     % data-dependent, so don't need to sample them with f*|f.
            # end
            # % reinstate randn state for further f sampling
            # randn('state', f_rng_state);
            #===================================================================
            #read_randoms(should=f_star_mean, true_random_source=False) # DEBUG
            #read_randoms(should=np.vstack(marginals_f).ravel(order='F'), true_random_source=False) # DEBUG
            
            # using marginals_f, compute current error and current LL test data
            current_error = compute_error_nlm(marginals_f)[0] # discard neg log marg for prediction from single f
            #read_randoms(1, should=current_error, true_random_source=False) # DEBUG
            scaled_ll_test = ll_test(f_star_mean)
            #read_randoms(1, should=scaled_ll_test, true_random_source=False) # DEBUG
            
            with open(result_prefix + "marginals.bin", 'ab') as marginals_file:
                write_marginals(marginals_f, marginals_file)

            # read marginals from disk
            marginals_read = np.array([0]) # init to non-empty list
            with open(result_prefix + "marginals.bin", 'rb') as marginals_file:
                all_marginals = read_marginals(marginals_file)

            #compute        
            #    - avg_error: test set error averaged over all f*|D draws 
            #    - ll_test: log-likelihood of very last f*|D sample
            marginals_after_burnin = all_marginals[len(all_marginals)//3:]
            (avg_error, avg_nlm) = compute_error_nlm(average_marginals(marginals_after_burnin))
            logger.info(("ESS it %g -- " +
                        "scaled LL train = %.5g -- " +
                        "test set error | last f = %.5g -- " + 
                        "scaled LL test | last f = %.5g -- " + 
                        "test set error (avg over f's)= %.5g -- " +
                        "average per pixel negative log posterior marginals = %.5g") % 
                        (mcmc_step, 
                         current_ll,
                         current_error, 
                         scaled_ll_test,
                         avg_error,
                         avg_nlm
                         )
                        )
        mcmc_step += 1 # now ready for next iteration
                        
        # finally save results for this MCMC step (avg_error, avg_nlp unchanged from previous step in case no prediction occurred)
        with open(result_prefix + 'results.bin', 'ab') as results_file:
            last_results = np.array([current_ll, 
                     current_error, 
                     scaled_ll_test,
                     avg_error,
                     avg_nlm], dtype=learn_predict_gpstruct.dtype)
            last_results.tofile(results_file) # file format = row-wise array, shape #mcmc steps * 5 float32
        
        # save state in case we are interrupted
        with open(result_prefix + 'state.pickle', 'wb') as random_state_file:
            pickle.dump({'prng' : read_randoms.prng,
                         'current_f' : current_f,
                         'mcmc_step' : mcmc_step,
                         'current_ll' : current_ll,
                         'current_error' : current_error, 
                         'scaled_ll_test' : scaled_ll_test,
                         'avg_error' : avg_error,
                         'avg_nlm' : avg_nlm}, 
                         random_state_file)
        
# LATER
# - separate learning (write marginals to disk) 
#    from prediction (read, skipping burnin, applying extra thinning, and compute errors)
# - f* vs f* MAP
# - warm start from existing f's and marginals ? (useful when must run on time-limited jobs, like HPC or Fear (12h))

import scipy.sparse.csr
def kernel_linear_unary(X_train, X_test, lhp, noise_param):
    p = np.dot(X_train,X_test.T)
    if isinstance(p, scipy.sparse.csr.csr_matrix):
        p = p.toarray() # cos if using X_train sparse vector, p will be a csr_matrix -- incidentally in this case the resulting k_unary cannot be flattened, it will result in a (1,X) 2D matrix !
    k_unary = np.exp(lhp["unary"]) * np.array(p, dtype=learn_predict_gpstruct.dtype)
    if (noise_param == 0):
        return k_unary
    else:
        return k_unary + (noise_param) * np.eye(k_unary.shape[0])
    # to build block diag matrices there's scipy.linalg.block_diag

# cache the result of compute_kStarTKInv_unary
# compute_kStarTKInv_unary operates on a hashable version of np.arrays (whcih are otherwise non-hashable because mutable).
# the code for hashable is from http://machineawakening.blogspot.co.at/2011/03/making-numpy-ndarrays-hashable.html
# hashability is required for storing in the dictionary in the Memoization class
# compute_kStarTKInv_unary can therefore be Memoize'd. it needs to unwrap the ndarrays before operating on them (with linalg.solve)
# construction a bit complex cos wanted to isolate main code from this memoization business

def compute_kStarTKInv_unary(k_unary, k_star_unary):
    '''
    must compute S = K* ' K^-1, equivalent to S K = K*', ie K' S' = K*, ie K S' = K* (cos K sym)
    x=solve(a,b) returns x solution of ax=b
    so solve(K, K*) gives S', hence need to transpose the result 
    '''
    return np.linalg.solve(k_unary.unwrap(), k_star_unary.unwrap()).T

def hashable_compute_kStarTKInv_unary(k_unary, kStar_unaryT):
    return compute_kStarTKInv_unary(hashable(k_unary), hashable(kStar_unaryT))

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

compute_kStarTKInv_unary = Memoize(compute_kStarTKInv_unary)

import hashlib
class hashable(object): # from http://machineawakening.blogspot.co.at/2011/03/making-numpy-ndarrays-hashable.html
    r'''Hashable wrapper for ndarray objects.

        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.

        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    '''
    def __init__(self, wrapped, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.

            wrapped
                The wrapped ndarray.

            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight = tight
        self.__wrapped = np.array(wrapped) if tight else wrapped
        self.__hash = int(hashlib.sha1(wrapped.view(np.uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return np.all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.

            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped