import numpy as np
import numpy.testing
import scipy.io
import learn_predict

def sample_from_kernel(n_labels, lower_chol_K_unary, lower_chol_K_binary, read_randoms):
    """
    lower_chol_K_binary optional, if absent will only sample the unaries
    """
    n_x = lower_chol_K_unary.shape[0]
    if not(lower_chol_K_binary == None):
        f = numpy.zeros(n_x * n_labels + n_labels**2, dtype=learn_predict.learn_predict_gpstruct.dtype)
    else:
        f = numpy.zeros(n_x * n_labels, dtype=learn_predict.learn_predict_gpstruct.dtype)
    # write samples into f (expanded), using lower_chol_K_unary (compact, ie implicitly repeated block-diagonal) for each of nLabels
    # ie the unary kernel is reused to generate a section of f for each value of y_t
    for i in range(n_labels): # vectorize ?
        f[i*n_x:(i+1)*n_x] = np.dot(lower_chol_K_unary, read_randoms(n_x, 'n')) #np.random.randn(n_x))
    if not(lower_chol_K_binary == None):
        f[- n_labels**2:] = np.dot(lower_chol_K_binary, read_randoms(n_labels**2, 'n'))#np.random.randn(n_labels**2));
    return f

def ESS(f, logli, n_labels, lower_chol_K_unary, lower_chol_K_binary, read_randoms):
    nu =  sample_from_kernel(n_labels, lower_chol_K_unary, lower_chol_K_binary, read_randoms)
    u = read_randoms(1, 'u')
#    u = np.random.rand(1)
#    print("initial logli evaluation")
    log_y = numpy.log(u) + logli(f)
    read_randoms(1, should=log_y)
    read_randoms(f.shape[0], should=f)
    v = read_randoms(1, 'u')
    theta = v*2*numpy.pi
#    theta = np.random.rand(1)*2*numpy.pi
    theta_min = theta - 2*numpy.pi
    theta_max = theta
    while True:
        fp = f*np.cos(theta)+nu*np.sin(theta)
        #print("f : %s" % f.dtype)
        #print("nu : %s" % nu.dtype)

#        print("log li evaluation in loop")
        cur_log_like = logli(fp)
        read_randoms(1, should=cur_log_like )
        read_randoms(fp.shape[0], should=fp)
        if (cur_log_like > log_y):
            break
        if (theta < 0):
            theta_min = theta
        else:
            theta_max = theta
        v = read_randoms(1, 'u')
        theta = v*(theta_max - theta_min) + theta_min
#        theta = np.random.rand(1)*(theta_max - theta_min) + theta_min
    return (fp, cur_log_like)

if __name__ == "__main__":
    A = np.random.randint(1,10,[5,5])
    print(A)
    (fp, cur_log_like) = ESS(np.zeros(30), lambda f: f.sum(), 6, np.linalg.cholesky(np.dot(A,A.T)))
    print(fp)
