import numpy as np
import util

"""
     ported to Python from Iain Murray's Matlab code, Sebastien Bratieres August 2014
"""

def update_theta_simple(theta, ff, lik_fn, 
    Lfn, # lower chol 
    slice_width=10, 
    theta_Lprior = lambda _lhp_target : 0 if np.all(_lhp_target>np.log(0.1)) and np.all(_lhp_target<np.log(10)) else np.NINF
    ):
    """
    update to GP hyperparam. slice sample theta| f, data. Does not update f, so requires separate procedure to update f (eg elliptical slice sampling).
    """

    #L = Lfn(theta)

    # Slice sample theta|ff
    class particle:
        pass
    particle.pos = theta
    particle.ff = ff
    slice_fn = lambda pp, Lpstar_min : eval_particle_simple(pp, lik_fn, theta_Lprior, Lpstar_min, Lfn)
    slice_fn(particle, Lpstar_min = np.NINF)
    slice_sweep(particle, slice_fn, sigma = abs(slice_width), step_out = (slice_width > 0))
    return (particle.pos, particle.ff, particle.lik_fn_ff) # could return L !? so you don't have to recompute it ?


def eval_particle_simple(pp, lik_fn, theta_Lprior, Lpstar_min, 
    Lfn # lower chol
    ): # should not need to return particle, can modify in-place
    """
    pp modified in place
    L is a precomputed chol(Kfn(pp.pos))
    alternatively, Lfn is a function that will compute L
    """

    Ltprior = theta_Lprior(pp.pos)
    if Ltprior == np.NINF: # save time in case Ltprior is NINF, don't need to run Lfn
        #print('off slice cos prior limit hit')
        pp.Lpstar = Ltprior
        pp.on_slice = False
        return 
    L = Lfn(pp.pos) # L = lower chol K_theta
    Lfprior = -0.5 * pp.ff.T.dot(L.solve_cholesky_lower(pp.ff)) - L.diag_log_sum(); # + const # this is log p(f|theta)
    # log(p(f|theta)) = log(N(pp.ff ; 0, U_theta)) = -1/2 f.T (L.T.dot(L))^1 f - log(sqrt(2 * pi * det(L.T.dot(L)))) 

    pp.lik_fn_ff = lik_fn(pp.ff)
    pp.Lpstar = pp.lik_fn_ff + Lfprior + Ltprior # log p(x|f) + log p(f|theta) + log p(theta)
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    #pp.L = L
    
# PRIOR WHITENING METHOD

def update_theta_aux_chol(theta, ff, lik_fn, 
    Lfn, # lower chol
    slice_width=10, 
    theta_Lprior = lambda _lhp_target : np.log((np.all(_lhp_target>np.log(0.1)) and np.all(_lhp_target<np.log(10))))
    ):
    """
    update to GP hyperparam. Fixes nu used to draw f, rather than f itself
    """

    L = Lfn(theta)
    nu = L.solve_triangular(ff)

    # Slice sample theta|nu
    class particle:
        pass
    particle.pos = theta
    slice_fn = lambda pp, Lpstar_min : eval_particle_aux_chol(pp, nu, lik_fn, theta_Lprior, Lpstar_min, Lfn)
    slice_fn(particle, Lpstar_min = np.NINF)
    slice_sweep(particle, slice_fn, sigma = abs(slice_width), step_out = (slice_width > 0))
    return (particle.pos, particle.ff, particle.lik_fn_ff)

def eval_particle_aux_chol(pp, nu, lik_fn, theta_Lprior, Lpstar_min, 
    Lfn # lower chol
    ): # should not need to return particle, can modify in-place
    """
    pp modified in place
    L is a precomputed chol(Kfn(pp.pos))
    alternatively, Lfn is a function that will compute L
    """

    Ltprior = theta_Lprior(pp.pos)
    if Ltprior == np.NINF:
        pp.on_slice = False
        pp.Lpstar = np.NINF
        return
    L = Lfn(pp.pos)
    ff = L.dot(nu)

    pp.lik_fn_ff = lik_fn(ff)
    pp.Lpstar = Ltprior + pp.lik_fn_ff
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    pp.L = L
    pp.ff = ff

def slice_sweep(particle, slice_fn, sigma=1, step_out=True):# should not need to return particle, can modify in-place
# %SLICE_SWEEP one set of axis-aligned slice-sampling updates of particle.pos
# %
# %     particle = slice_sweep(particle, slice_fn[, sigma[, step_out]])
# %
# % The particle position is updated with a standard univariate slice-sampler.
# % Stepping out is linear (if step_out is true), but shrinkage is exponential. A
# % sensible strategy is to set sigma conservatively large and turn step_out off.
# % If it's hard to set a good sigma though, you should leave step_out=true.
# %
# % Inputs:
# %     particle   sct   Structure contains:
# %                              .pos - initial position on slice as Dx1 vector
# %                                     (or any array)
# %                           .Lpstar - log probability of .pos (up to a constant)
# %                         .on_slice - needn't be set initially but is set
# %                                     during slice sampling. Particle must enter
# %                                     and leave this routine "on the slice".
# %     slice_fn   @fn   particle = slice_fn(particle, Lpstar_min)
# %                      If particle.on_slice then particle.Lpstar should be
# %                      correct, otherwise its value is arbitrary.
# %        sigma (D|1)x1 step size parameter(s) (default=1)
# %     step_out   1x1   if non-zero, do stepping out procedure (default), else
# %                      only step in (saves on fn evals, but takes smaller steps)
# %
# % Outputs:
# %     particle   sct   particle.pos and .Lpstar are updated.

#% Originally based on pseudo-code in David MacKay's text book p375
#% Iain Murray, May 2004, January 2007, June 2008, January 2009
# Sebastien Bratieres ported to Python August 2014

#    DD = particle.pos.shape[0] # dimensionality of parameter space
#if length(sigma) == 1
#    sigma = repmat(sigma, DD, 1);
#end
# Note: in Iain's code, sigma can be an array of step-sizes, aligned with particle.pos which is the array theta. In my code, theta is a dict. I haven't ported the feature allowing sigma to be an array. So here, the step-size is equal for all hyperparameters.
    import util
    assert(particle.on_slice)
    # A random order (in hyperparameters) is more robust generally and important inside algorithms like nested sampling and AIS
    for (dd, x_cur) in enumerate(particle.pos):
    #for dd in np.random.permutation(particle.pos.shape[0]):
    #    x_cur = particle.pos[dd]
        #print('working in param %s' % d)
        # Lpstar_min is sampled so that pstar_min ~ U[0,pstar]
        Lpstar_min = particle.Lpstar + np.log(util.read_randoms(1, 'u')[0]) # take [0] to make scalar
        #print('particle.on_slice? %g' % particle.on_slice)
        # % Create a horizontal interval (x_l, x_r) enclosing x_cur
        rr = util.read_randoms(1, 'u')[0]
        x_l = x_cur - rr*sigma
        x_r = x_cur + (1-rr)*sigma
        if step_out:
            #print('stepping out left with Lpstar_min=%g' % Lpstar_min)
            particle.pos[dd] = x_l
            while True:
                slice_fn(particle, Lpstar_min)
                #print('on-slice %g is (particle.Lpstar = %g >= Lpstar_min = %g)' % (particle.on_slice, particle.Lpstar, Lpstar_min))
                if not particle.on_slice:
                    break
                particle.pos[dd] = particle.pos[dd] - sigma
            #print('placed x_l = %g, now stepping out right' % x_l)
            x_l = particle.pos[dd]
            particle.pos[dd] = x_r
            while True:
                slice_fn(particle, Lpstar_min)
                if not particle.on_slice:
                    break
                particle.pos[dd] = particle.pos[dd] + sigma
            #print('placed x_r = %g, particle.on_slice? %g' % (x_r, particle.on_slice))

            x_r = particle.pos[dd]

        #% Make proposals and shrink interval until acceptable point found
        #% One should only get stuck in this loop forever on badly behaved problems,
        #% which should probably be reformulated.
        while True:
            particle.pos[dd] = util.read_randoms(1, 'u')[0]*(x_r - x_l) + x_l # proposed new point
            slice_fn(particle, Lpstar_min)
            if particle.on_slice:
                break # Only way to leave the while loop.
            else:
                # Shrink in, from the left or the right according to which side the proposal is
                if particle.pos[dd] > x_cur:
                    x_r = particle.pos[dd]
                elif particle.pos[dd] < x_cur:
                    x_l = particle.pos[dd]
                else:
                    print("back to current position %f (x_cur==particle.pos[d])" % x_cur)
                    print("Lpstar_min = %.15g" % Lpstar_min)
                    # check that current position is on slice
                    particle.pos[dd] = x_cur
                    slice_fn(particle, Lpstar_min)
                    print("particle on slice? %s" % str(particle.on_slice))
                    raise Exception('BUG DETECTED: Shrunk to current position and still not acceptable.')
                    

# SURROGATE DATA METHOD

def update_theta_aux_surr(theta, ff, lik_fn, Kfn, theta_Lprior, slice_width=10, aux=0.1):
# %UPDATE_THETA_AUX_SURR MCMC update to GP hyper-param based on aux. noisy vars
# %
# %     [theta, ff] = update_theta_aux_noise(theta, ff, lik_fn, Kfn, aux, theta_Lprior);
# %
# % Inputs:
# %             theta Kx1 hyper-parameters (can be an array of any size)
# %                ff Nx1 apriori Gaussian values
# %               lik_fn @fn Log-likelihood function, lik_fn(ff) returns a scalar
# %               Kfn @fn Kfn(theta) returns NxN covariance matrix
# %                       NB: this should contain jitter (if necessary) to
# %                       ensure the result is positive definite.
# %
# % Specify aux in one of three ways:
# % ---------------------------------
# %           aux_std Nx1 std-dev of auxiliary noise to add to each value
# %                       (can also be a 1x1).
# % OR
# %        aux_std_fn @fn Function that returns auxiliary noise level(s) to use:
# %                       aux_std = aux_std_fn(theta, K);
# % OR
# %               aux cel A pair: {aux_std_fn, aux_cache} called like this:
# %                       [aux_std, aux_cache] = aux_std_fn(theta, K, aux_cache);
# %                       The cache could be used (for example) to notice that
# %                       relevant parts of theta or K haven't changed, and
# %                       immediately returning the old aux_std.
# % ---------------------------------
# %
# %      theta_Lprior @fn Log-prior, theta_Lprior(theta) returns a scalar
# %
# % Outputs:
# %             theta Kx1 updated hyper-parameters (Kx1 or same size as inputted)
# %                ff Nx1 updated apriori Gaussian values
# %               aux  -  Last aux_std computed, or {aux_std_fn, aux_cache},
# %                       depending on what was passed in.
# %             cholK NxN chol(Kfn(theta))
# %
# % The model is draw g ~ N(0, K + S), (imagine as f ~ N(0, K) + noise with cov S)
# % Draw f ~ N(m_p, C_p), using posterior mean and covariance given g.
# % But implement that using nu ~ randn(N,1). Then clamp nu's while changing K.
# %
# % K is obtained from Kfn.
# % S = diag(aux_std.^2), or for scalar aux_std (aux_std^2 * eye(N)).

# % Iain Murray, November 2009, January 2010, May 2010

# % If there is a good reason for it, there's no real reason full-covariance
# % auxiliary noise couldn't be added. It would just be more expensive as sampling
# % would require decomposing the noise covariance matrix. For now this code
# % hasn't implemented that option.

    class pp:
        pass
    pp.pos = theta
    pp.Kfn = Kfn
    # only implementing fixed auxiliary noise level for now
    pp.aux_std = aux
    pp.aux_var = aux * aux
    # if isnumeric(aux)
        # % Fixed auxiliary noise level
        # pp.adapt_aux = 0;
        # pp.aux_std = aux;
        # pp.aux_var = aux.*aux;
    # elseif iscell(aux)
        # % Adapting noise level, with computations cached
        # pp.adapt_aux = 2;
        # pp.aux_fn = aux{1};
        # pp.aux_cache = aux{2};
    # else
        # % Simple function to choose noise level
        # pp.adapt_aux = 1;
        # pp.aux_fn = aux;
    # end
    pp.gg = np.zeros_like(ff)
    theta_changed(pp)

    # Instantiate g|f
    pp.gg = ff + util.read_randoms(len(ff), 'u') * pp.aux_std
    pp.Sinv_g = pp.gg / pp.aux_var

    # Instantiate nu|f,gg
	# Algo 3, line 2: eta = L_R_theta^-1 (f - m_theta,g)
	# Matlab: x=A'\b <=> A' x = b
	# Numpy: A' x = b <=> x = np.linalg.solve(A.T, b) 
	# my gram_compact implementation: A' x = b <=> x = A.T_solve(b)
    pp.nu = pp.U_invR.dot(ff) - pp.U_invR.T_solve(pp.Sinv_g) #pp.U_invR*ff(:) - pp.U_invR'\pp.Sinv_g;


    # Slice sample update of theta|g,nu
    slice_fn = lambda pp, Lpstar_min : eval_particle_aux_surr(pp, Lpstar_min, lik_fn, theta_Lprior)
    # Compute current log-prob (up to constant) needed by slice sampling:
    eval_particle_aux_surr(pp, np.NINF, lik_fn, theta_Lprior, theta_unchanged = True) # theta hasn't moved yet, don't recompute covariances
    assert(pp.on_slice)
    if False:
        print("after eval_particle_aux_surr")
        print('particle on slice? %g, pp.Lpstar = %g' % (pp.on_slice, pp.Lpstar))
    # would want to replace by 
    # slice_fn(particle, Lpstar_min = np.NINF)
    # if it weren't for theta_unchanged=True which is hard to pass

    slice_sweep(pp, slice_fn, sigma = abs(slice_width), step_out = (slice_width > 0))
    #print(pp.pos)
    #print(pp.lik_fn_ff)
    return (pp.pos, pp.ff, pp.lik_fn_ff)
    
    # optional outputs, not doing yet
    # if iscell(aux)
        # aux = {pp.aux_fn, pp.aux_cache};
    # else
        # aux = pp.aux_std;
    # end
    # cholK = pp.U;


def theta_changed(pp):
    """ Will call after changing hyperparameters to update covariances and their decompositions."""
    theta = pp.pos;
    #print(theta)
    K = pp.Kfn(theta); # TODO for v1, cannot use compact representation
    #print(K.gram_unary[:5,:5])
    # if pp.adapt_aux
        # if pp.adapt_aux == 1
            # pp.aux_std = pp.aux_fn(theta, K);
        # elseif pp.adapt_aux == 2
            # [pp.aux_std, pp.aux_cache] = pp.aux_fn(theta, K, pp.aux_cache);
        # end
        # pp.aux_var = pp.aux_std .* pp.aux_std;
        # pp.Sinv_g = pp.gg ./ pp.aux_var;
    # end
    pp.U = K.cholesky_T() # Matlab chol(K);
    #print(pp.U.gram_unary[:5,:5])
	# Matlab chol(K) = upper Cholesky = lower Cholesky transpose = np.linalg.cholesky(K).T
    # pp.iK = inv(K)
    pp.U_invR = K.inv_add_diag_cholesky_T(1/pp.aux_var) # Matlab chol(plus_diag(pp.iK, 1./pp.aux_var));
	# %pp.U_noise = chol(plus_diag(K, aux_var_vec));
    assert_triu(pp.U)
    assert_triu(pp.U_invR)

def assert_triu(a):
    import numpy.testing
    numpy.testing.assert_almost_equal(a.gram_unary, np.triu(a.gram_unary))
    
def eval_particle_aux_surr(pp, Lpstar_min, lik_fn, theta_Lprior, theta_unchanged = False):
    #print("calling eval particle, theta_unchanged=%g" % theta_unchanged)
    # Prior on theta
    Ltprior = theta_Lprior(pp.pos)
    #    print(Ltprior)
    if Ltprior == np.NINF:
        pp.on_slice = False
        pp.Lpstar = np.NINF
        return
    if not theta_unchanged:
        theta_changed(pp)

    # Update f|gg,nu,theta
    pp.ff = pp.U_invR.solve(pp.nu) + pp.U_invR.T().solve_cholesky_lower(pp.Sinv_g) # pp.U_invR\pp.nu + solve_chol(pp.U_invR, pp.Sinv_g);

    # % Compute joint probability and slice acceptability.
    # % I have dropped the constant: -0.5*length(pp.gg)*log(2*pi)
    # %Lgprior = -0.5*(pp.gg'*solve_chol(pp.U_noise, pp.gg)) - sum(log(diag(pp.U_noise)));
    # %pp.Lpstar = Ltprior + Lgprior + lik_fn(pp.ff);
    # %
    # % This version doesn't need U_noise, but commenting out the U_noise line and
    # % using this version doesn't actually seem to be faster?
    Lfprior = -0.5 * pp.ff.T.dot(pp.U.T().solve_cholesky_lower(pp.ff)) - pp.U.diag_log_sum() #-0.5*(pp.ff'*solve_chol(pp.U, pp.ff)) - sum(log(diag(pp.U)));
    LJacobian = - pp.U_invR.diag_log_sum() # -sum(log(diag(pp.U_invR)));
#    %LJacobian = sum(log(diag(pp.U_R)));
    Lg_f = -0.5 * np.sum((pp.gg - pp.ff) ** 2) / pp.aux_var - 0.5 * np.sum(np.log(pp.aux_var * np.ones(pp.ff.shape[0]))) #-0.5*sum((pp.gg - pp.ff).^2)./pp.aux_var - sum(log(pp.aux_var.*ones(size(pp.ff))));
    pp.lik_fn_ff = lik_fn(pp.ff)
    pp.Lpstar = Ltprior + Lg_f + Lfprior + pp.lik_fn_ff + LJacobian
    # log p(theta) + log p(g|f) + log p(f|theta) + log p(x|f) + log p()
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    if False:
        print('particle on slice? %g, pp.Lpstar = %g, Lpstar_min = %g. Ltprior = %g, Lg_f=%g, Lfprior=%g, lik_fn_ff=%g, LJacobian=%g. pp.U.diag_log_sum()=%g' % (pp.on_slice, pp.Lpstar, Lpstar_min, Ltprior, Lg_f, Lfprior, pp.lik_fn_ff, LJacobian, pp.U.diag_log_sum()))
        #return np.array([pp.Lpstar, Ltprior, Lg_f, Lfprior, pp.lik_fn_ff, LJacobian])

        
import kernels
import gc
def update_theta_aux_surr_new(theta, ff, lik_fn, Kfn, theta_Lprior, n_train, n_labels, logger, slice_width=10, aux=0.1):
    class pp:
        pass
    pp.pos = theta
    aux_var = aux * aux
    identity = lambda __aux_var : kernels.gram_compact.identity(n_train, n_labels, __aux_var)
    
    # sample g|f from N(g|f, S)
    g = ff + identity(np.sqrt(aux_var)).dot(util.read_randoms(len(ff), 'n'))
    # note identity(np.sqrt(aux_var)) = chol(S)
    
    # compute ancillaries of theta
    logger.debug('compute ancillaries for theta')
    ancillaries_theta_fn = lambda theta : ancillaries_theta_complete(theta, Kfn, identity(aux_var), g)
    ancillaries_theta = ancillaries_theta_fn(theta)

    # construct eta
    logger.debug('construct eta')
    eta = ancillaries_theta.chol_R.solve_triangular(ff) \
        + ancillaries_theta.chol_R.T().dot(identity(1/aux_var).dot(g))# requires solve_triangular, S.inv
    # OPTIMIZE can certainly optimize Sinv.dot(g) when S = alpha* I
    # note identity(1/aux_var) = Sinv
    
    # set particle
    pp.g = g
    pp.pos = theta
    pp.ancillaries_theta = ancillaries_theta
    pp.f = ff
    # initial evaluation of Lpstar and on-sliceness
    logger.debug('initial evaluation of Lpstar and on-sliceness')
    eval_particle_aux_surr_new(pp, np.NINF, lik_fn, theta_Lprior, ancillaries_theta_fn, theta_unchanged = True) # theta hasn't moved yet, don't recompute covariances
    assert(pp.on_slice)
    if False:
        print("after eval_particle_aux_surr")
        print('particle on slice? %g, pp.Lpstar = %g' % (pp.on_slice, pp.Lpstar))

    # update theta by slice sampling (1 sweep)
    logger.debug('slice sampling sweep')
    slice_fn = lambda pp, Lpstar_min : eval_particle_aux_surr_new(pp, Lpstar_min, lik_fn, theta_Lprior, ancillaries_theta_fn)
    slice_sweep(pp, slice_fn, sigma = abs(slice_width), step_out = (slice_width > 0))

    # update f from new theta, ancillaries(new theta), g, eta
    logger.debug('update f')
    m = pp.ancillaries_theta.K.dot(pp.ancillaries_theta.Zinv_g)
    pp.f = pp.ancillaries_theta.chol_R.dot(eta) + m # requires from gram_compact: minus matrix, dot matrix, solve_chol matrix
    pos = pp.pos
    f = pp.f
    lik_fn_ff = pp.lik_fn_ff
    del pp
    gc.collect()
    return (pos, f, lik_fn_ff)

def eval_particle_aux_surr_new(pp, Lpstar_min, lik_fn, theta_Lprior, ancillaries_theta_fn, theta_unchanged = False):
    Ltprior = theta_Lprior(pp.pos) # compute p(theta)
    #    print(Ltprior)
    if Ltprior == np.NINF:
        pp.on_slice = False
        pp.Lpstar = np.NINF
        return

    pp.lik_fn_ff = lik_fn(pp.f) # compute p(f|D)

    if not theta_unchanged: # recompute ancillaries_theta
        pp.ancillaries_theta = ancillaries_theta_fn(pp.pos)
    
    # compute p(g|theta)
    log_g_theta = - pp.ancillaries_theta.chol_Z.diag_log_sum() - 0.5 * pp.g.T.dot(pp.ancillaries_theta.Zinv_g) 
    # compute p(theta | eta, g, D) \propto p(f|D) p(g|theta) p(theta)
    pp.Lpstar = pp.lik_fn_ff + log_g_theta + Ltprior
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    
def ancillaries_theta_complete(theta, Kfn, S, g):
    class ancillaries_theta:
        pass
    ancillaries_theta.K = Kfn(theta)
    ancillaries_theta.chol_Z = (ancillaries_theta.K + S).cholesky()
    ancillaries_theta.chol_R = (S - S.dot(ancillaries_theta.chol_Z.solve_cholesky_lower(S))).cholesky()
    del S # order of previous statements important; so that I can de-reference S as soon as possible (since this will impact the max mem usage)
    gc.collect()
    ancillaries_theta.Zinv_g = ancillaries_theta.chol_Z.solve_cholesky_lower(g)
    return ancillaries_theta

"""
OPTIMIZING SURROGATE DATA
• can carry out further algebraic simplifications entirely in Python (unit tests use only Python code)
	• extract gram_compact and *_aux_surr into own file v1
	• start doing transformations to code: v2. in unit test, check that results and if possible intermediate results are identical to v1. inform the transformations with algebra. write down algebra and derivations in script, no latex. try reduce the number of methods used in gram_compact.
	• the last working version will be duplicated, minus the unit tests, into resp. gram_compact.py and slice_spl_hp.py
• TODO other performance tricks
	-  avoid recomputing p everytime we recompute the unary kernel, cache! to do that, must somehow identify which of (train, train) (train, trest) (test, test) we've been asked to compute -- how ?? prepapre the 3 types on function initialization ? -- however, this is not useful for the ARD kernel, cos there we can't avoid recomputing the kernel on each point
	- cache nu for a given (f, kernel) (nu was known inside ESS or inside sample from kernel)

"""