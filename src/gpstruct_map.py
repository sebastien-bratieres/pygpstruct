import chain_forwards_backwards_logsumexp as fwdbwd

def sufficient_statistics(data_train, Y):
    """ input: Y contains labellings for all y^(n)
    returns counts, over the entire dataset, of the form $ \sum_n \#c\in(\mathbf{x}^{(n)},\mathbf{y^{(n)}} $
    """
    grad_f = np.zeros((data_train.n_labels * data_train.X.shape[0] + data_train.n_labels **2), dtype=util.dtype_for_arrays)
    # noting grad_f, but this is one of the terms of grad_f, really
    for n in range(data_train.N):
        for t in range(data_train.object_size[n]):
            grad_f[data_train.unaries[n][t, Y[n][t]]] += 1
        for t in range(1, data_train.object_size[n]):
            grad_f[data_train.binaries[Y[n][t-1], Y[n][t]]] += 1
    return grad_f

#def expected_statistics(data_train, f):
#    """ computes the expected count of average statistics, ie the term d log Z/df. this is one of the two terms of
#    d log lik (D_train) / df"""
#    n_samples = 50 # computing expectation over fixed number of samples -- surely there are cleverer schemes
#    accumulated_statistics = np.zeros_like(f, dtype=util.dtype_for_arrays)
#    for n in range(n_samples):
#        accumulated_statistics += sufficient_statistics(data_train, sample_given_f(data_train, f))
#    return accumulated_statistics/n_samples

def expected_statistics(dataset, f):
    grad_f_contribution = np.zeros_like(f, dtype=util.dtype_for_arrays)
    log_edge_pot = f[dataset.binaries]
    for n in range(dataset.N):
        log_node_pot = f[dataset.unaries[n]]
        grad_f_contribution[dataset.unaries[n]] += np.exp(
            fwdbwd.compute_log_gamma_normalized(log_edge_pot, log_node_pot, dataset.object_size[n], dataset.n_labels, 
                                                                                   fwdbwd.log_alpha, fwdbwd.log_beta, fwdbwd.temp_array_1, fwdbwd.temp_array_2, fwdbwd.temp_array_3)
            )
        log_ksi = np.exp(
            fwdbwd.compute_log_ksi_normalized(log_edge_pot, log_node_pot, dataset.object_size[n], dataset.n_labels, 
                                                                                   fwdbwd.log_alpha, fwdbwd.log_beta, fwdbwd.temp_array_1, fwdbwd.temp_array_2)
            )
        grad_f_contribution[dataset.binaries] += log_ksi.sum(axis=0) # collapsing along t dimension, result is shape (n_labels, n_labels): expectation for each binary factor over entire sequence
    return grad_f_contribution
        
def grad_likelihood(data_train, f):
    return sufficient_statistics(data_train, data_train.Y) - expected_statistics(data_train, f)

def compute_Kinv_f(f, L):
    return L.solve_cholesky_lower(f) # K^-1 f = (L L^T)^-1 f

def grad_prior(f, L):
    return -L.solve_cholesky_lower(f)

def grad_posterior(f, L, data_train):
    return grad_likelihood(data_train, f) + grad_prior(f, L)

import gc
def func(x, L, ll_train, logger):
    x = util.dtype_for_arrays(x) # sometimes x will come in float64
    log_prior_val = util.dtype_for_arrays(-1/2) * x.dot(compute_Kinv_f(x, L)) # log p(f) = -1/2 f^T K^-1 f + const (which we disregard)
    log_posterior_val = log_prior_val + ll_train(x)
    logger.debug("objective function value %.5g" % (- log_posterior_val))
    gc.collect()
    return -log_posterior_val

def grad(x, L, data_train, logger):
    x = util.dtype_for_arrays(x) # sometimes x will come in float64
    return - grad_posterior(x, L, data_train)

import logging
import time
import sys
def obtain_logger(config):
    logger = logging.getLogger('pyGPstruct') # reuse logger if existing
    logger.setLevel(2)
    logger.propagate = False
    logger.handlers = [] # remove any existing handlers
    # create formatter
    formatter = logging.Formatter('%(asctime)sZ - %(levelname)s - %(message)s')
    formatter.converter=time.gmtime # will use GMT timezone for output, hence the Z
    # console log handler
    ch = logging.StreamHandler(stream=sys.stdout)
    #ch.setLevel(logging.INFO) # avoid clogging console (more info in file)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if 'result_prefix' in config:
        fh = logging.FileHandler(config['result_prefix'] + "log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

import scipy
import kernels
import numpy as np
import pickle
import util
import os

def run(config):
    np.random.seed = config["seed"]
    # prepare data-dependent functions
    n_features = {'basenp' : 6438, 'chunking' : 29764, 'segmentation' : 1386, 'japanesene' : 102799}[config['task']]
    n_data = {'basenp' : 300, 'chunking' : 100, 'segmentation' : 36, 'japanesene' : 100}[config['task']]
    n_data_train = {'basenp' : 150, 'chunking' : 50, 'segmentation' : 20, 'japanesene' : 50}[config['task']]
    fold_indices = np.loadtxt(config['pygpstruct_location'] + '/data/datasplit.n_data=%s.txt' % n_data, dtype=np.int16) - 1 # need -1 because doing +1 inside prepare_data_chain

    common_arguments = {
    'data_indices_train' : fold_indices[config['fold'],:n_data_train],
    'data_indices_test' : fold_indices[config['fold'],n_data_train:],
    'data_folder' : config['pygpstruct_location'] + '/data/' + config['task'],
    'native_implementation': True,
    }

    import dataset_chain
    data_train = dataset_chain.dataset_chain(task = config['task'], data_folder=common_arguments['data_folder'], sub_indices=common_arguments['data_indices_train'])
    data_test = dataset_chain.dataset_chain(task = config['task'], data_folder=common_arguments['data_folder'], sub_indices=common_arguments['data_indices_test'])
    # will need to use inside prepare_from_data.posterior_marginals
    fwdbwd.preassign(max_T = max(data_train.object_size.max(), data_test.object_size.max()), n_labels = data_train.n_labels)

    os.makedirs(config['result_prefix'], exist_ok=True)
    logger=obtain_logger(config)

    import prepare_from_data_chain
    (ll_train,
    posterior_marginals_test,
    compute_error_nlm,
    ll_test,
    average_marginals,
    write_marginals,
    read_marginals,
    n_labels,
    X_train,
    X_test) = prepare_from_data_chain.prepare_from_data_chain(task=config['task'],
                                                           logger=logger,
                                                           **common_arguments
                                               )

    # prepare K, L=cholK, and then func and grad for the minimizer
    #kernel = kernels.kernel_exponential_ard
    #lhp = {'unary': np.log(1), 'binary': np.log(1), 'jitter': np.log(1e-4), 'variances' : +1 * np.ones((n_features))}
    kernel = kernels.kernel_linear
    lhp = {'unary': np.log(1), 'binary': np.log(1), 'jitter': np.log(1e-4)}
    L = kernels.compute_lower_chol_k(kernel, lhp, X_train, n_labels)
    func_L = lambda x : func(x,L, ll_train, logger)
    grad_L = lambda x : grad(x,L, data_train, logger)

    # how to choose x0 ?
    if False:
        # read off stored experiment results
        with open('/xvdb/results-2015-06-23/pygpstruct_50k.task=%(task)s.hp_sampling_mode=None.parameterization=ub.fold=%(fold)g.hp_sampling_thinning=1000/state.pickle'
                  % config, 'rb') as file:
            x0 = pickle.load(file)['current_f']
    else:
        # pick random position
        u = np.random.randn(data_train.n_labels * data_train.X.shape[0] + data_train.n_labels **2)
        x0 = L.dot(u.astype(util.dtype_for_arrays))

    res = scipy.optimize.minimize(func_L, x0, method=config['optimization.method'],
                                  jac=grad_L, options=config['optimization.options'])
    f_star_mean = kernels.compute_k_star_T_k_inv(kernel, lhp, X_train, X_test, n_labels, L).dot(res.x)
    marginals_f_star = posterior_marginals_test(f_star_mean)
    error_metrics = compute_error_nlm(marginals_f_star)

    result = {'f' : res.x,
                         'accuracy' : error_metrics[0],
                         'nlm' : error_metrics[1]}
    result.update(config)
    with open(config['result_prefix'] + 'map.pickle', 'wb') as result_file:
        pickle.dump(result, result_file)
    logger.info(result)


# TESTS
if __name__ == "__main__":

    # TEST setup, copied from run() ------------
    config = {'task': 'segmentation', 'fold':0, 'pygpstruct_location': '/xvdb/pygpstruct/'}

    n_features = {'basenp' : 6438, 'chunking' : 29764, 'segmentation' : 1386, 'japanesene' : 102799}[config['task']]
    n_data = {'basenp' : 300, 'chunking' : 100, 'segmentation' : 36, 'japanesene' : 100}[config['task']]
    n_data_train = {'basenp' : 150, 'chunking' : 50, 'segmentation' : 20, 'japanesene' : 50}[config['task']]
    fold_indices = np.loadtxt(config['pygpstruct_location'] + '/data/datasplit.n_data=%s.txt' % n_data, dtype=np.int16) - 1 # need -1 because doing +1 inside prepare_data_chain

    common_arguments = {
    'data_indices_train' : fold_indices[config['fold'],:n_data_train],
    'data_indices_test' : fold_indices[config['fold'],n_data_train:],
    'data_folder' : config['pygpstruct_location'] + '/data/' + config['task'],
    'native_implementation': True,
    }

    import dataset_chain
    data_train = dataset_chain.dataset_chain(task = config['task'], data_folder=common_arguments['data_folder'], sub_indices=common_arguments['data_indices_train'])
    data_test = dataset_chain.dataset_chain(task = config['task'], data_folder=common_arguments['data_folder'], sub_indices=common_arguments['data_indices_test'])
    # will need to use inside prepare_from_data.posterior_marginals

    logger=obtain_logger(config)
    logger.setLevel(logging.INFO)

    import prepare_from_data_chain
    (ll_train,
    posterior_marginals_test,
    compute_error_nlm,
    ll_test,
    average_marginals,
    write_marginals,
    read_marginals,
    n_labels,
    X_train,
    X_test) = prepare_from_data_chain.prepare_from_data_chain(task=config['task'],
                                                           logger=logger,
                                                           **common_arguments
                                               )

    # TEST
    # suppose all f's corresponding to binary factors are equal, say 0.
    # then these two values should be equal:
    # - #edges in data
    # - expectation of #all edges over dataset
    f = np.zeros(data_train.n_labels * data_train.X.shape[0] + data_train.n_labels **2, dtype=util.dtype_for_arrays)

    fwdbwd.preassign(max_T = data_train.object_size.max(), n_labels = data_train.n_labels)
    np.testing.assert_almost_equal((data_train.n_points - data_train.N),
                                   expected_statistics(data_train, f)[data_train.binaries].sum())

    # prepare K, L=cholK, and then func and grad for the minimizer
    #kernel = kernels.kernel_exponential_ard
    #lhp = {'unary': np.log(1), 'binary': np.log(1), 'jitter': np.log(1e-4), 'variances' : +1 * np.ones((n_features))}
    kernel = kernels.kernel_linear
    lhp = {'unary': np.log(1), 'binary': np.log(1), 'jitter': np.log(1e-4)}
    L = kernels.compute_lower_chol_k(kernel, lhp, X_train, n_labels)
    func_L = lambda x : func(x,L, ll_train, logger)
    grad_L = lambda x : grad(x,L, data_train, logger)

    u = np.random.randn(data_train.n_labels * data_train.X.shape[0] + data_train.n_labels **2)
    x0 = L.dot(u.astype(util.dtype_for_arrays))
    # END TEST setup, copied from run() ------------

    # TEST: manual gradient check in one direction
    # I can do a manual check in one single direction, to have an idea of orders of magnitude: take a step in direction eps * np.ones(), and compare:

    eps = 1e-12
    np.testing.assert_almost_equal(func_L(x0 + eps) - func_L(x0), # first terms of Taylor expansion
                                   grad_L(x0).dot(eps*np.ones_like(x0)), # gradient value
                                   decimal=3)
    # gradient works ! at least in that direction.

    # TEST: gradient check
    # check_grad will make a move of amplitude eps into each direction (ie along each dimension of x0), and compute the difference between the finite-diff approx (f(x + eps) - f(x) / eps), and the gradient in that direction
    assert(scipy.optimize.check_grad(func_L, grad_L, x0, epsilon=0.1) < 1e5)
    # this is long, because it iterates over the number of dimensions of f-space, eg 3522 in the present configuration