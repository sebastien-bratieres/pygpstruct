import numba
import numpy as np
import numpy.testing

@numba.njit
def lse_numba(a):
    result = 0.0
    max_a = np.max(a)
    for i in range(a.shape[0]):
        result += np.exp(a[i] - max_a)
    return np.log(result) + max_a

@numba.njit('double[:](double[:,:], double[:])') 
def lse_numba_axis_1(a, result):
    for r in range(a.shape[0]):
        result[r] = lse_numba(a[r,:])
    return result


@numba.njit
def sum_vec(a,b, result):
    for r in range(a.shape[0]):
        result[r] = a[r]+b[r]
    return result

@numba.njit
def lse_numba_axis_1_tile(a,v, 
    result, # (n_labels)
    temp_array): # (n_labels)
    for r in range(a.shape[0]):
        #result[r] = lse_numba(a[r,:] + v) # slower
        result[r] = lse_numba(sum_vec(a[r,:], v, temp_array))
    return result

@numba.njit
def lse_numba_2d(a):
    result = 0.0
    max_a = np.max(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result += np.exp(a[i,j] - max_a)
    return np.log(result) + max_a

@numba.jit
def compute_log_gamma_normalized(log_edge_pot, #'(t-1,t)', 
                  log_node_pot, # '(t, label)', 
                  T, 
                  n_labels,
                  log_alpha,
                  log_beta,
                  temp_array_1, 
                  temp_array_2, 
                  temp_array_3): 
    """ to obtain the (smoothed) posterior marginals p(y_t = j | X_1:T) = gamma_t(j) """

    log_alpha = compute_log_alpha(log_edge_pot, log_node_pot, T, n_labels, log_alpha, temp_array_1, temp_array_2)
    log_beta = compute_log_beta(log_edge_pot, log_node_pot, T, n_labels, log_beta, temp_array_1, temp_array_2)
    log_gamma = log_alpha[:T, :] + log_beta[:T, :]     # reduce shape of arrays (was max_T long, now is T long). will ensure log_gamma has the right shape, and might require fewer calculations
    # perform the following (ie normalize log_gamma with axis=1=, but faster (cos removing np.tile python call):
    #log_gamma -= np.tile(lse_numba_axis_1(log_gamma), (n_labels, 1)).T
    temp_array_3 = lse_numba_axis_1(log_gamma, temp_array_3) # must use array_3 here, shape (max_T)
    for c in range(n_labels):
        log_gamma[:,c] -= temp_array_3[:T] # assign only up to index T (shape temp_array_3 : (max_T))
    return log_gamma #return p(z_t = j| x_1:T) ie normalized gamma, ref MLAPP (17.52), shape (t, label)

@numba.njit
def compute_log_alpha(log_edge_pot, #'(t-1,t)', 
                  log_node_pot, # '(t, label)', 
                  T, 
                  n_labels, 
                  log_alpha, temp_array_1, temp_array_2):
    """ to obtain Z 
    - log_edge_pot = bold \psi in MLAPP (17.48)
    - log_node_pot[t,j] = \psi_t(y_t = j)
    
    wrt MLAPP, I also note the evidence X, and I note the labels Y (MLAPP notes Z)
    """

    log_alpha[0,:] = log_node_pot[0,:]
# don't need to preserve normalizing constant, except to produce log-likelihood
#    log_kappa = np.empty((T))
#    log_kappa[0] = lse_numba(log_alpha[0,:])
#    log_alpha[0,:] -= log_kappa[0]
    log_alpha[0,:] -= lse_numba(log_alpha[0,:])
    for t in range(1,T):
        log_alpha[t,:] = log_node_pot[t,:] + lse_numba_axis_1_tile(log_edge_pot.T, log_alpha[t-1,:], temp_array_1, temp_array_2)
#        log_kappa[t] = lse_numba(log_alpha[t,:])
#        log_alpha[t,:] -= log_kappa[t]
        log_alpha[t,:] -= lse_numba(log_alpha[t,:])
    return log_alpha

@numba.njit
def compute_log_beta(log_edge_pot, log_node_pot, T, n_labels, 
    log_beta, temp_array_1, temp_array_2):
    # set log_beta[T-1,:] to 0
    log_beta[T-1,:] = 0
# no need to preserve kappa
#    log_kappa = np.empty((T))
#    log_kappa[T-1] = lse_numba(log_beta[-1,:])
    #log_beta[-1,:] -= log_kappa[-1] # not necessary cos already normalized
    for t in range(1,T):
        log_beta[T-1-t,:] = lse_numba_axis_1_tile(log_edge_pot, log_node_pot[T-t,:] + log_beta[T-t,:], temp_array_1, temp_array_2)
#        log_kappa[T-1-t] = lse_numba(log_beta[T-1-t,:])
        log_beta[T-1-t,:] -= lse_numba(log_beta[T-1-t,:])
    return log_beta
    
@numba.jit
def compute_log_Z(log_edge_pot, #'(t-1,t)', 
                  log_node_pot, # '(t, label)', 
                  T, 
                  n_labels,
                  log_alpha,
                  temp_array_1, 
                  temp_array_2): 
    log_alpha[0,:] = log_node_pot[0,:]
    log_Z = lse_numba(log_alpha[0,:]) # this is log_kappa[0]
    log_alpha[0,:] -= log_Z # normalize alpha
    for t in range(1,T):
        log_alpha[t,:] = log_node_pot[t,:] + lse_numba_axis_1_tile(log_edge_pot.T, log_alpha[t-1,:], temp_array_1, temp_array_2)
        #print("pre_mult" + str(np.exp(lse_numba_axis_1_tile(log_edge_pot.T, log_alpha[t-1,:], temp_array_1, temp_array_2))))
        log_kappa_t = lse_numba(log_alpha[t,:])  # this is log_kappa[t]
        log_alpha[t,:] -= log_kappa_t # normalize alpha
        log_Z += log_kappa_t # at the end, log_Z = sum(log_kappa[:])
    return log_Z
    
#@numba.jit # faster without !
def compute_log_ksi_normalized(log_edge_pot, #'(t-1,t)', 
                  log_node_pot, # '(t, label)', 
                  T, 
                  n_labels,
                  log_alpha,
                  log_beta,
                  temp_array_1, 
                  temp_array_2): 
    """ to obtain the two-slice posterior marginals p(y_t = i, y_t+1 = j| X_1:T) = normalized ksi_t,t+1(i,j) """
    # in the following, will index log_ksi only with t, to stand for log_ksi[t,t+1]. including i,j: log_ksi[t,i,j]

    log_alpha = compute_log_alpha(log_edge_pot, log_node_pot, T, n_labels, log_alpha, temp_array_1, temp_array_2)
    log_beta = compute_log_beta(log_edge_pot, log_node_pot, T, n_labels, log_beta, temp_array_1, temp_array_2)
    log_ksi = np.empty((T-1, n_labels, n_labels))
    for t in range(T-1):
        psi_had_beta = log_node_pot[t+1,:] + log_beta[t+1, :] # represents psi_t+1 \hadamard beta_t+1 in MLAPP eq 17.67
        log_ksi[t,:,:] = log_edge_pot 
        for c in range(n_labels):
            for d in range(n_labels):
                log_ksi[t,c,d] += log_alpha[t,d] + psi_had_beta[c]
        # normalize current ksi[t,:,:] over both dimensions. This is not required of ksi, strictly speaking, but the output of the function needs to be normalized, and it's cheaper to do it in-place on ksi than to create a fresh variable to hold the normalized values
        log_ksi[t,:,:] -= lse_numba_2d(log_ksi[t,:,:])
    return log_ksi #return p(z_t = j| x_1:T) ie normalized gamma, ref MLAPP (17.52), shape (t, label)
    
# pre-assigning these arrays for speed.
# CATCH: Numba seems to access globals more slowly than local variables. Therefore it is preferable to pass the pointers to these arrays in each function call that needs them, despite the typing overhead.
def preassign(max_T, n_labels):
    global log_alpha, log_beta, temp_array_1, temp_array_2, temp_array_3
    log_alpha = np.empty((max_T, n_labels))
    log_beta = np.empty((max_T, n_labels))
    temp_array_1 = np.empty((n_labels))
    temp_array_2 = np.empty((n_labels))
    temp_array_3 = np.empty((max_T))

# pygpstruct_location = '/xvdb/pygpstruct'

# %load_ext autoreload
# %autoreload 2

# import numpy as np
# import sys
# sys.path.append(pygpstruct_location + '/src/') # replace by your path to .py files

# import chain_forwards_backwards_logsumexp as lse

# n_labels = 10
# max_T = 20
# T = 10
# log_node_pot = np.random.rand(T, n_labels)
# log_edge_pot = np.random.rand(n_labels, n_labels)
# dataset_Y_n = np.random.randint(low=0, high=n_labels, size=(T))

# lse.preassign(max_T, n_labels)

# %%timeit
# lse.compute_log_Z(log_edge_pot,
                             # log_node_pot,
                             # object_size,
                             # n_labels,
                             # lse.log_alpha, # pre-assigned memory space to speed up likelihood computation
                             # lse.temp_array_1,
                             # lse.temp_array_2)

# %%timeit 
# lse.compute_log_gamma(log_edge_pot,
                             # log_node_pot,
                             # T,
                             # n_labels,                             
              # lse.log_alpha, lse.log_beta, 
              # lse.temp_array_1,                             lse.temp_array_2)

              
# OLD CODE



# @numba.njit
# def forwards_algo_log_alpha(log_edge_pot, #'(t-1,t)', 
                  # log_node_pot, # '(t, label)', 
                  # T, 
                  # n_labels):
    # """ to obtain Z 
    # - log_edge_pot = bold \psi in MLAPP (17.48)
    # - log_node_pot[t,j] = \psi_t(y_t = j)
    
    # wrt MLAPP, I also note the evidence X, and I note the labels Y (MLAPP notes Z)
    # """

    # temp_array_1 = np.empty((n_labels))
    # temp_array_2 = np.empty((n_labels))
    # log_alpha = np.empty((T, n_labels))
    # log_alpha[0,:] = log_node_pot[0,:]
# # don't need to preserve normalizing constant, except to produce log-likelihood
# #    log_kappa = np.empty((T))
# #    log_kappa[0] = lse_numba(log_alpha[0,:])
# #    log_alpha[0,:] -= log_kappa[0]
    # log_alpha[0,:] -= lse_numba(log_alpha[0,:])
    # for t in range(1,T):
        # log_alpha[t,:] = log_node_pot[t,:] + lse_numba_axis_1_tile(log_edge_pot.T, log_alpha[t-1,:], temp_array_1, temp_array_2)
# #        log_kappa[t] = lse_numba(log_alpha[t,:])
# #        log_alpha[t,:] -= log_kappa[t]
        # log_alpha[t,:] -= lse_numba(log_alpha[t,:])
    # return log_alpha

# @numba.jit
# def forwards_algo_log_Z(log_edge_pot, #'(t-1,t)', 
                  # log_node_pot, # '(t, label)', 
                  # T, 
                  # n_labels,
                  # log_alpha,
                  # log_kappa,
                  # temp_array_1, 
                  # temp_array_2): 
    # # log_alpha and log_kappa are pre-assigned for speed. The T dimension is set to max_T over the entire train and test datasets. Must not access beyond index T in this function !
    
# #    log_alpha = np.empty((T, n_labels))
# #    log_kappa = np.empty((T))
    # log_alpha[0,:] = log_node_pot[0,:]
    # log_kappa[0] = lse_numba(log_alpha[0,:])
    # log_alpha[0,:] -= log_kappa[0] # normalize alpha
    # for t in range(1,T):
        # log_alpha[t,:] = log_node_pot[t,:] + lse_numba_axis_1_tile(log_edge_pot.T, log_alpha[t-1,:], temp_array_1, temp_array_2)
        # #print("pre_mult" + str(np.exp(lse_numba_axis_1_tile(log_edge_pot.T, log_alpha[t-1,:], temp_array_1, temp_array_2))))
        # log_kappa[t] = lse_numba(log_alpha[t,:])
        # log_alpha[t,:] -= log_kappa[t] # normalize alpha
    # log_Z = 0
    # for i in range(T):
        # log_Z += log_kappa[i]
        # #print(log_kappa[i])
    # return log_Z

# @numba.jit
# def forwards_backwards_algo_log_gamma(log_edge_pot, log_node_pot, T, n_labels):
    # """ to obtain the (smoothed) posterior marginals p(y_t = j | X_1:T) = gamma_t(j) """

    # temp_array_1 = np.empty((n_labels))
    # temp_array_2 = np.empty((n_labels))
    # temp_array_3 = np.empty((T))
    # log_alpha = forwards_algo_log_alpha(log_edge_pot, log_node_pot, T, n_labels)
    # log_beta = np.empty((T, n_labels))
    
    # # set log_beta[T-1,:] to 0
    # # need loop cos the following won't work
# #    log_beta[T-1,:] = np.zeros((n_labels)) # numba cannot coerce
    # for c in range(n_labels):
        # log_beta[T-1,c] = 0
# # no need to preserve kappa
# #    log_kappa = np.empty((T))
# #    log_kappa[T-1] = lse_numba(log_beta[-1,:])
    # #log_beta[-1,:] -= log_kappa[-1] # not necessary cos already normalized
    # for t in range(1,T):
        # log_beta[T-1-t,:] = lse_numba_axis_1_tile(log_edge_pot, log_node_pot[T-t,:] + log_beta[T-t,:], temp_array_1, temp_array_2)
# #        log_kappa[T-1-t] = lse_numba(log_beta[T-1-t,:])
        # log_beta[T-1-t,:] -= lse_numba(log_beta[T-1-t,:])
    
    # log_gamma = log_alpha + log_beta
    # # perform the following (ie normalize log_gamma with axis=1=, but faster (cos removing np.tile python call):
    # #log_gamma -= np.tile(lse_numba_axis_1(log_gamma), (n_labels, 1)).T
    # temp = lse_numba_axis_1(log_gamma, temp_array_3)
    # for c in range(n_labels):
        # log_gamma[:,c] -= temp
    # return log_gamma #return gamma MLAPP (17.52), shape (t, label)

# def preassign(max_T, n_labels):
    # global log_alpha, log_beta, log_kappa, temp_array_1, temp_array_2
    # log_alpha = np.empty((max_T, n_labels))
    # log_beta = np.empty((max_T, n_labels))
    # log_kappa = np.empty((max_T, n_labels))
    # temp_array_1 = np.empty((n_labels))              
