import numpy as np
import os
import scipy.sparse
import chain_forwards_backwards_logsumexp
import prepare_from_data
import learn_predict
import kernels
import numba
import copy
import util

try:
    import chain_forwards_backwards_native
    native_implementation_found = True
    @numba.jit # not even sure this speeds anything up. if it doesn't speed up anything, could remove this function and just define it 
    # inline as a lambda function
    def log_likelihood_function_native(node_pot, edge_pot, dataset_Y_n, object_size, n_labels):
        # native implementation takes linear (not log) domain potentials
        return chain_forwards_backwards_native.log_likelihood(edge_pot, node_pot, dataset_Y_n)
except ImportError as ie:
    native_implementation_found = False

def default_set_lhp_target(lhp_, lhp_target):
    new_lhp = copy.deepcopy(lhp_)
    new_lhp["unary"] = lhp_target[0]
    new_lhp["binary"] = lhp_target[1]
    return new_lhp
def default_get_lhp_target(lhp_):
    return np.array([lhp_["unary"], lhp_["binary"]], dtype=util.dtype_for_arrays)
    
def learn_predict_gpstruct_wrapper(
    data_indices_train=np.arange(0,10), 
    data_indices_test=np.arange(10,20),
    task='basenp',
    data_folder=None,
    result_prefix='/tmp/pygpstruct/',
    console_log=True, # log to console as well as to file ?
    n_samples=0, 
    prediction_thinning=1, # how often (in terms of MCMC iterations) to carry out prediction, ie compute f*|f and p(y*)
    prediction_verbosity=None,
    lhp_init={'unary': np.log(1), 'binary': np.log(0.01), 'jitter' : np.log(1e-4)},
    lhp_gset = (default_get_lhp_target, default_set_lhp_target),
    lhp_prior = lambda _lhp_target : 0 if (np.all(_lhp_target>np.log(1e-3)) and np.all(_lhp_target<np.log(1e2))) else np.NINF,
    hp_sampling_thinning=1, 
    hp_sampling_mode=None,
    kernel=kernels.kernel_linear,
    random_seed=0,
    stop_check=None,
    native_implementation=False,
    log_f=False,
    no_hotstart = False
    ):
    if data_folder==None:
        data_folder = './data/%s' % task
        
    learn_predict.learn_predict_gpstruct(lambda logger : 
                       prepare_from_data_chain(
                            task=task,
                            data_indices_train=data_indices_train, 
                            data_indices_test=data_indices_test,
                            data_folder=data_folder,
                            logger=logger,
                            native_implementation=native_implementation
                            ), 
                       result_prefix=result_prefix,
                       console_log=console_log,
                       n_samples=n_samples, 
                       prediction_thinning=prediction_thinning, 
                       prediction_verbosity=prediction_verbosity,
                       hp_sampling_thinning=hp_sampling_thinning, 
                       hp_sampling_mode=hp_sampling_mode,
                       lhp_init=lhp_init,
                       lhp_gset=lhp_gset,
                       lhp_prior=lhp_prior,
                       kernel=kernel,
                       random_seed=random_seed,
                       stop_check=stop_check,
                       log_f=log_f,
                       no_hotstart=no_hotstart
                       )

def prepare_from_data_chain(task, data_indices_train, data_indices_test, data_folder, logger, native_implementation):
    logger.info("prepare_from_data_chain started with arguments: " + str(locals()))
    
    import dataset_chain
    data_train = dataset_chain.dataset_chain(task, data_folder, data_indices_train)
    data_test = dataset_chain.dataset_chain(task, data_folder, data_indices_test)
    logger.debug("loaded data from disk")
    
    # pre-assigning for speed so that there's no memory assingment in the most inner loop of the likelihood computation
    max_T = np.max([data_train.object_size.max(), data_test.object_size.max()])
    
    if native_implementation:
        if native_implementation_found:
            chain_forwards_backwards_native.init_kappa(max_T)
            return (
                lambda f : prepare_from_data.log_likelihood_dataset(f, data_train, log_likelihood_function_native, logger, ll_fun_wants_log_domain=False),  # that's ll_fun_wants_log_domain=False
                lambda f : prepare_from_data.posterior_marginals(f, data_test, marginals_function), 
                lambda marginals : compute_error_nlm(marginals, data_test),
                lambda f : prepare_from_data.log_likelihood_dataset(f, data_test, log_likelihood_function_native, logger, ll_fun_wants_log_domain=False), # that's ll_fun_wants_log_domain=False
                prepare_from_data.average_marginals, 
                write_marginals,
                lambda marginals_file : read_marginals(marginals_file, data_test),
                data_train.n_labels, data_train.X, data_test.X) 
        else:
            raise Exception("You have set native_implementation=True, but there has been an ImportError on import chain_forwards_backwards_native, and so I can't find the native implementation.")
    else:
        log_likelihood_function_numba.log_alpha = np.empty((max_T, data_train.n_labels))
        log_likelihood_function_numba.log_kappa = np.empty((max_T))    
        log_likelihood_function_numba.temp_array_1 = np.empty((data_train.n_labels))
        log_likelihood_function_numba.temp_array_2 = np.empty((data_train.n_labels))
        return (
            lambda f : prepare_from_data.log_likelihood_dataset(f, data_train, log_likelihood_function_numba, logger, ll_fun_wants_log_domain=True),
            lambda f : prepare_from_data.posterior_marginals(f, data_test, marginals_function), 
            lambda marginals : compute_error_nlm(marginals, data_test),
            lambda f : prepare_from_data.log_likelihood_dataset(f, data_test, log_likelihood_function_numba, logger, ll_fun_wants_log_domain=True),
            prepare_from_data.average_marginals, 
            write_marginals,
            lambda marginals_file : read_marginals(marginals_file, data_test),
            data_train.n_labels, data_train.X, data_test.X) 

@numba.jit
def log_likelihood_function_numba(log_node_pot, log_edge_pot, dataset_Y_n, object_size, n_labels):
        # log-likelihood for point n, for observed data dataset.Y[n], consists of 
        # "numerator": sum of potentials for  binaries (indexed (y_{t-1}, y_{t})) and unaries (indexed (t, y))
        log_pot = log_edge_pot[dataset_Y_n[:-1], dataset_Y_n[1:]].sum() + log_node_pot[np.arange(object_size), dataset_Y_n].sum()
        # using array indexing in numpy 
        # nodes: for each position in the chain (np.arange(object_size)), select the correct unary factor corresponding to y (dataset_Y_n)
        # edges: select edge factors with tuples (y_{t-1}, y_{t}). Use as the first index, y_{t-1}, obtained by cutting off dataset_Y_n before the last position; and as the second index, y{t}, obtained by shifting dataset_Y_n to the left (ie [1:]).
        
        # "denominator": log_Z obtained from forwards pass
        log_Z = chain_forwards_backwards_logsumexp.forwards_algo_log_Z(log_edge_pot,
                                     log_node_pot,
                                     object_size,
                                     n_labels,
                                     log_likelihood_function_numba.log_alpha, # pre-assigned memory space to speed up likelihood computation
                                     log_likelihood_function_numba.log_kappa,
                                     log_likelihood_function_numba.temp_array_1,
                                     log_likelihood_function_numba.temp_array_2)
        return (log_pot - log_Z)

def marginals_function(log_node_pot, log_edge_pot, object_size, n_labels):
    """
    marginals returned have shape (object_size, n_labels)
    """
    return np.exp(chain_forwards_backwards_logsumexp.forwards_backwards_algo_log_gamma(log_edge_pot, log_node_pot, object_size, n_labels))

def write_marginals(marginals_f, marginals_file):
    #print(marginals_f)
    np.array(np.vstack(marginals_f), dtype=util.dtype_for_arrays).tofile(marginals_file) # can hstack cos all elements have #labels rows
    
def read_marginals(marginals_file, dataset):
    result = np.fromfile(marginals_file, dtype=util.dtype_for_arrays)
    result = result.reshape((-1, dataset.n_points * dataset.n_labels))
    result = np.split(result, result.shape[0], axis=0) # make list of rows
    # each element now contains a 1D array containing all the marginals for all data points. need to turn that into a list of n elements.
    result = [np.split(marginals_f.reshape((-1, dataset.n_labels)), dataset.object_size.cumsum()[:-1], axis=0) for marginals_f in result] # undo the hstack followed by writing to disk in row-first C order: first reshape 1D array so that each row is for one label; then split row-wise so that each block corresponds to an object
    # dataset.object_size.cumsum()[:-1] => row indices where to split the marginals array
    #print(result)
    return result
    
def compute_error_nlm(marginals, dataset):
    stats_per_object = np.empty((dataset.N,2)) # first col for error rate, second col for neg log marg
    for n, marginals_n in enumerate(marginals):
        #print("marginals_n.shape: " + str(marginals_n.shape))
        ampm = np.argmax(marginals_n, axis = 1) # argmax posterior marginals
        #print("comparing %s to %s" % (str(ampm.shape), str(dataset.Y[n].shape)))
        stats_per_object[n,0] = (ampm != dataset.Y[n]).sum()
        
        # from marginals, use dataset.Y[n] to select on axis "labels"
        avg_nlpm_object = np.empty_like(dataset.Y[n])
        T = dataset.Y[n].shape[0]
        for t in range(T):
            avg_nlpm_object[t] = -np.log(marginals_n[t, dataset.Y[n][t]])
        stats_per_object[n,1] = avg_nlpm_object.sum()
    return stats_per_object.sum(axis=0) / dataset.n_points # per point averaging
    
if __name__ == "__main__":
    learn_predict_gpstruct_wrapper()
