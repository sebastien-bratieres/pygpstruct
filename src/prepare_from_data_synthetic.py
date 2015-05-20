import numba
@numba.jit
def lse_numba(a):
    result = 0.0
    largest_in_a = 0.0
    for i in range(a.shape[0]): # numba is slow when using max or np.max, so re-implementing
        if (a[i] > largest_in_a):
            largest_in_a = a[i]
    for i in range(a.shape[0]):
        result += np.exp(a[i] - largest_in_a)
    return np.log(result) + largest_in_a

@numba.jit
def log_likelihood(log_node_pot, log_edge_pot, dataset_Y_n, object_size, n_labels):
    return log_node_pot[0, dataset_Y_n[0]] - lse_numba(log_node_pot[0,:])

@numba.jit
def marginals_function(log_node_pot, log_edge_pot, object_size, n_labels):
    return np.exp(log_node_pot[0,:] - lse_numba(log_node_pot[0,:])).reshape((1,n_labels))

import prepare_from_data_chain
import prepare_from_data
def prepare_from_data_single_label(logger=None, n_data=100, debug=False, synthetic_data_type='5-class', noise_feature_weight=1):
    class dataset:
        pass
    dataset.N = n_data
    dataset.object_size = np.ones((dataset.N), dtype=np.int)
    dataset.n_points = dataset.N
    dataset.Y = []
    dataset.unaries = []
    if synthetic_data_type=='5-class':
        dataset.n_labels = 5
        n_features = dataset.n_labels
        dataset.X = np.zeros((dataset.n_points, n_features))
        for n in range(dataset.N):
            dataset.Y.append(np.array([np.mod(n, dataset.n_labels) ])) 
            dataset.X[n, dataset.Y[n][0]] = 1 # only feature set to 1 is the indicator of the class, easy to learn
    elif (synthetic_data_type == 'single feature'):
        dataset.n_labels = 2
        n_features = 1
        dataset.X = np.zeros((dataset.n_points, n_features))
        for n in range(dataset.N//2):
            dataset.Y.append(np.array([0])) #np.mod(n, dataset.n_labels) ]) #numpy.random.randint(dataset.n_labels)], dtype=np.int)
            dataset.X[n, 0] = np.array([0]) #dataset.Y[n]
        for n in range(dataset.N//2, dataset.N):
            dataset.Y.append(np.array([1])) #np.mod(n, dataset.n_labels) ]) #numpy.random.randint(dataset.n_labels)], dtype=np.int)
            dataset.X[n, 0] = np.array([1]) #dataset.Y[n]
    elif (synthetic_data_type == '5-class with noise features'):
        dataset.n_labels = 5
        n_features = dataset.n_labels * 2
        dataset.X = np.zeros((dataset.n_points, n_features))
        np.random.seed(0)
        for n in range(dataset.N):
            dataset.Y.append(np.array([np.mod(n, dataset.n_labels) ])) 
            dataset.X[n, dataset.Y[n][0]] = 1 
            dataset.X[n, dataset.n_labels:] = np.random.rand(dataset.n_labels)*noise_feature_weight # 5 last features are noise
    elif (synthetic_data_type[0] == 'predefined'):
        dataset.n_labels = synthetic_data_type[1].n_labels
        dataset.X = synthetic_data_type[1].X
        dataset.Y = synthetic_data_type[1].Y
    else:
        raise Exception('you mistyped the synthetic data set you want')

    for n in range(dataset.N):
        dataset.unaries.append(np.zeros((dataset.object_size[n], dataset.n_labels), dtype=np.int))

    f_index_max = 0 # contains largest index in f
    for yt in range(dataset.n_labels):
        for n in range(dataset.N):
            for t in range(dataset.object_size[n]):
                dataset.unaries[n][t, yt] = f_index_max
                f_index_max = f_index_max + 1

    dataset.binaries = np.arange(f_index_max, f_index_max + dataset.n_labels**2).reshape((dataset.n_labels, dataset.n_labels), order='F')
    dataset.f_index_max = f_index_max + dataset.n_labels**2 

    
    data_train = dataset
    data_test = dataset

    #print(dataset.X.tolist())
    #print(dataset.unaries)
    #print(dataset.binaries)
    #logger.debug("dataset object: " + str(vars(dataset)))
    if debug:
        return dataset
    else:
        return (
            lambda f : prepare_from_data.log_likelihood_dataset(f, data_train, log_likelihood, logger, ll_fun_wants_log_domain=True),
            lambda f : prepare_from_data.posterior_marginals(f, data_test, marginals_function), 
            lambda marginals : prepare_from_data_chain.compute_error_nlm(marginals, data_test),
            lambda f : prepare_from_data.log_likelihood_dataset(f, data_test, log_likelihood, logger, ll_fun_wants_log_domain=True),
            prepare_from_data.average_marginals, 
            prepare_from_data_chain.write_marginals,
            lambda marginals_file : prepare_from_data_chain.read_marginals(marginals_file, data_test),
            dataset.n_labels, data_train.X, data_test.X) 
    
import learn_predict
import kernels
import numpy as np
import copy
def learn_predict_gpstruct_wrapper(
    data_folder=None,
    result_prefix='/tmp/pygpstruct/',
    start_from_clean_slate=False,
    n_samples=100, 
    prediction_thinning=1, # how often (in terms of MCMC iterations) to carry out prediction, ie compute f*|f and p(y*)
    prediction_verbosity=None,
    hp_sampling_thinning=1,
    hp_sampling_mode=None,
    lhp_init={'jitter':np.log(1e-4), 'unary':np.log(1), 'binary':np.log(0.01), 'variances' : np.log(np.ones((10), dtype=learn_predict.dtype_for_arrays))},
    kernel=kernels.kernel_exponential_ard,
    n_data = 100,
    synthetic_data_type = '5-class',
    noise_feature_weight=1,
    console_log=True,
    random_seed=0,
    ):        

    if start_from_clean_slate:
        import shutil
        shutil.rmtree(result_prefix)
    
    def set_lhp_target(lhp_, lhp_target):
        new_lhp = copy.deepcopy(lhp_)
        new_lhp["variances"] = lhp_target
        return new_lhp
    def get_lhp_target(lhp_):
        return copy.deepcopy(lhp_["variances"])

    learn_predict.learn_predict_gpstruct(lambda logger : 
        prepare_from_data_single_label(
            n_data=n_data,
            logger=logger,
            synthetic_data_type = synthetic_data_type,
            noise_feature_weight=noise_feature_weight
            ), 
        result_prefix=result_prefix,
        console_log=console_log,
        n_samples=n_samples, 
        prediction_thinning=prediction_thinning, 
        prediction_verbosity=prediction_verbosity,
        hp_sampling_thinning=hp_sampling_thinning,
        hp_sampling_mode=hp_sampling_mode,
        lhp_gset = (get_lhp_target, set_lhp_target),
        lhp_init=lhp_init,
        kernel=kernel,
        random_seed=random_seed,
        )

