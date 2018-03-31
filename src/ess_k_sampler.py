import numpy as np
import numpy.testing
import scipy.io
import learn_predict
import util

def ESS(f, logli, lower_chol_K_compact, read_randoms):
    nu =  lower_chol_K_compact.dot(read_randoms(len(f), 'n')) # sample from K
    u = read_randoms(1, 'u')
    # initial log lik evaluation
    log_y = np.log(u) + logli(f)
    v = read_randoms(1, 'u')
    theta = v*2*np.pi
    theta_min = theta - 2*np.pi
    theta_max = theta
    
    stopwatch = util.stop_check(delay = 300) 
    while not stopwatch.evaluate():
        fp = f*np.cos(theta)+nu*np.sin(theta)
        # log lik evaluation in loop
        cur_log_like = logli(fp)
        if (cur_log_like > log_y):
            break
        if (theta < 0):
            theta_min = theta
        else:
            theta_max = theta
        v = read_randoms(1, 'u')
        theta = v*(theta_max - theta_min) + theta_min
    if not stopwatch.evaluate(): # normal termination condition
        return (fp, cur_log_like)
    else: # time out termination
        return ESS(f, logli, lower_chol_K_compact, read_randoms) # in this case, start ESS procedure from scratch, ie recurse (we are confident this is not going to happen very often, if at all, hence the recursion will terminate)