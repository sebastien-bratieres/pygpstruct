import numpy as np

def posterior_marginals(f, dataset, marginals_function):
    """
    def posterior_marginals(f : "learnt latent variables, shape column", 
                        dataset : "dataset object") -> "list of (normalized) posterior marginals, length N":
    """
    pm = []
    for n in range(dataset.N):
        log_node_pot = f[dataset.unaries[n]]
        log_edge_pot = f[dataset.binaries]
        posterior_marginals_single = marginals_function(log_node_pot, log_edge_pot, dataset.object_size[n], dataset.n_labels) # in grid case, object_size will be ignored
        pm.append(posterior_marginals_single)
    return pm

def average_marginals(marginals_list):
    number_marginals = len(marginals_list)
    if (number_marginals == 1):
        return marginals_list[0]
    else:
        averaged_marginals = []
        N = len(marginals_list[0]) # says how many objects there are
        for n in range(N): # assumes all lists of marginals are the same length, namely N
            marginals_object_n = [marginals_all_objects[n] for marginals_all_objects in marginals_list]
            averaged_marginals.append(reduce(lambda x,y : add_marginals(x,y), marginals_object_n) / number_marginals) # mean of marginals for object n
        return averaged_marginals
        
def add_marginals(x,y):
    """
    x,y : marginals for one object
    adds marginals, and returns their sum
    """
    return x+y
    
import numba
#@numba.jit           
def ll_scaled_fun(f, dataset, log_likelihood_function, f_in_log_domain):
    """
    f : log-domain potentials
    f_in_log_domain : whether or not to pass log-domain f to the log-likelihood function
    """
    #print("f.dtype : %s" % f.dtype)
    if not f_in_log_domain:
        f = np.exp(f) # changing semantics of f instead of inserting if's on edge_pot=... and node_pot=...
    ll = 0
    edge_pot = f[dataset.binaries]
#    print(dataset.binaries)
#    print(log_edge_pot)
    #assert(log_edge_pot.shape == (dataset.n_labels, dataset.n_labels))
    for n in range(dataset.N):
        node_pot = f[dataset.unaries[n]]
#        print(dataset.unaries[n][:5,:5,0])
#        print(log_node_pot[:5,:5,0])
        #assert(log_node_pot.shape == (dataset.object_size, dataset.object_size, dataset.n_labels))
        
        ll += log_likelihood_function(node_pot, edge_pot, dataset.Y[n], dataset.object_size[n], dataset.n_labels) 
        # in grid case, object_size will be ignored
        # potentials will be log if f_in_log_domain=True (grid), otherwise they are in linear domain (chain)
        if (ll >0):
            print('ll now : %g' % ll)
            print(log_node_pot.tolist())
            print(log_edge_pot.tolist())
            print(dataset.Y)
            print(dataset)
    return ll #/dataset.number_points
