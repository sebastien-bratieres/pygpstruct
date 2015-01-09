import numpy as np
import util

"""
     ported to Python from Iain Murray's Matlab code, Sebastien Bratieres August 2014
"""

def update_theta_simple(theta, ff, Lfn, Ufn, 
    slice_width=10, 
    theta_Lprior = lambda _lhp_target : 0 if np.all(_lhp_target>np.log(0.1)) and np.all(_lhp_target<np.log(10)) else np.NINF
    ):
    """
    update to GP hyperparam. slice sample theta| f, data. Does not update f, so requires separate procedure to update f (eg elliptical slice sampling).
    """

    U = Ufn(theta)

    # Slice sample theta|ff
    class particle:
        pass
    particle.pos = theta
    particle.ff = ff
    slice_fn = lambda pp, Lpstar_min : eval_particle_simple(pp, Lfn, theta_Lprior, Lpstar_min, Ufn)
    slice_fn(particle, Lpstar_min = np.NINF)
    slice_sweep(particle, slice_fn, sigma = abs(slice_width), step_out = (slice_width > 0))
    return (particle.pos, particle.ff, particle.Lfn_ff) # could return U !? so you don't have to recompute it ?


def eval_particle_simple(pp, Lfn, theta_Lprior, Lpstar_min, Ufn): # should not need to return particle, can modify in-place
    """
    pp modified in place
    U is a precomputed chol(Kfn(pp.pos))
    alternatively, Ufn is a function that will compute U
    """

    Ltprior = theta_Lprior(pp.pos)
    if Ltprior == np.NINF: # save time in case Ltprior is NINF, don't need to run Ufn
        #print('off slice cos prior limit hit')
        pp.Lpstar = Ltprior
        pp.on_slice = False
        return 
    U = Ufn(pp.pos)
    L_inv_f = U.T_solve(pp.ff) # equivalent of L.solve. NB all my U variables are actually lower triangular and should be renamed!
    Lfprior = -0.5 * L_inv_f.T.dot(L_inv_f) - U.diag_log_sum(); # + const 
    # log(p(f|theta)) = log(N(pp.ff ; 0, U_theta)) = -1/2 f.T (U.T.dot(U))^1 f - log(sqrt(2 * pi * det(U.T.dot(U)))) 

    pp.Lfn_ff = Lfn(pp.ff)
    pp.Lpstar = pp.Lfn_ff + Lfprior + Ltprior # p(x|f) + p(f|theta) + p(theta)
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    pp.U = U
    
def update_theta_aux_chol(theta, ff, Lfn, Ufn, 
    slice_width=10, 
    theta_Lprior = lambda _lhp_target : np.log((np.all(_lhp_target>np.log(0.1)) and np.all(_lhp_target<np.log(10))))
    ):
    """
    update to GP hyperparam. Fixes nu used to draw f, rather than f itself
    """

    U = Ufn(theta)
    nu = U.T_solve(ff)

    # Slice sample theta|nu
    class particle:
        pass
    particle.pos = theta
    slice_fn = lambda pp, Lpstar_min : eval_particle_aux_chol(pp, nu, Lfn, theta_Lprior, Lpstar_min, Ufn)
    slice_fn(particle, Lpstar_min = np.NINF)
    slice_sweep(particle, slice_fn, sigma = abs(slice_width), step_out = (slice_width > 0))
    return (particle.pos, particle.ff, particle.Lfn_ff)

def eval_particle_aux_chol(pp, nu, Lfn, theta_Lprior, Lpstar_min, Ufn): # should not need to return particle, can modify in-place
    """
    pp modified in place
    U is a precomputed chol(Kfn(pp.pos))
    alternatively, Ufn is a function that will compute U
    """

    Ltprior = theta_Lprior(pp.pos)
    if Ltprior == np.NINF:
        pp.on_slice = False
        pp.Lpstar = np.NINF
        return
    U = Ufn(pp.pos)
    ff = U.T_dot(nu) # ff = np.dot(nu.T,U).T

    pp.Lfn_ff = Lfn(ff)
    pp.Lpstar = Ltprior + pp.Lfn_ff
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    pp.U = U
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
        #print('working in param %s' % d)
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
            #    print('on-slice %g is (particle.Lpstar = %g >= Lpstar_min = %g)' % (particle.on_slice, particle.Lpstar, Lpstar_min))
                if not particle.on_slice:
                    break
                particle.pos[dd] = particle.pos[dd] - sigma
            #print('placed x_l, now stepping out right')
            x_l = particle.pos[dd]
            particle.pos[dd] = x_r
            while True:
                slice_fn(particle, Lpstar_min)
                if not particle.on_slice:
                    break
                particle.pos[dd] = particle.pos[dd] + sigma
            #print('placed x_r, particle.on_slice? %g' % particle.on_slice)

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
                    print(x_cur)
                    print(particle.pos[dd])
                    print(x_cur == particle.pos[dd])
                    # check that current position is on slice
                    particle.pos[dd] = x_cur
                    slice_fn(particle, Lpstar_min)
                    print("particle on slice? %s" % str(particle.on_slice))
                    raise Exception('BUG DETECTED: Shrunk to current position and still not acceptable.')
                    

# SURROGATE DATA METHOD

def update_theta_aux_surr(theta, ff, Lfn, Kfn, theta_Lprior, slice_width=10, aux=0.1):
# %UPDATE_THETA_AUX_SURR MCMC update to GP hyper-param based on aux. noisy vars
# %
# %     [theta, ff] = update_theta_aux_noise(theta, ff, Lfn, Kfn, aux, theta_Lprior);
# %
# % Inputs:
# %             theta Kx1 hyper-parameters (can be an array of any size)
# %                ff Nx1 apriori Gaussian values
# %               Lfn @fn Log-likelihood function, Lfn(ff) returns a scalar
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
    pp.size_f = ff.shape[0]
    pp.gg = np.zeros((pp.size_f))
    theta_changed(pp)

    # Instantiate g|f
    pp.gg = ff + np.random.randn((pp.size_f)) * pp.aux_std
    pp.Sinv_g = pp.gg / pp.aux_var

    # Instantiate nu|f,gg
	# Algo 3, line 2: eta = L_R_theta^-1 (f - m_theta,g)
	# Matlab: x=A'\b <=> A' x = b
	# Numpy: A' x = b <=> x = np.linalg.solve(A.T, b) 
	# my gram_compact implementation: A' x = b <=> x = A.T_solve(b)
    pp.nu = pp.U_invR.dot(ff) - pp.U_invR.T_solve(pp.Sinv_g) #pp.U_invR*ff(:) - pp.U_invR'\pp.Sinv_g;


    # Slice sample update of theta|g,nu
    slice_fn = lambda pp, Lpstar_min : eval_particle_aux_surr(pp, Lpstar_min, Lfn, theta_Lprior)
    # Compute current log-prob (up to constant) needed by slice sampling:
    eval_particle_aux_surr(pp, np.NINF, Lfn, theta_Lprior, theta_unchanged = True) # theta hasn't moved yet, don't recompute covariances
    assert(pp.on_slice)
    if False:
        print("after eval_particle_aux_surr")
        print('particle on slice? %g, pp.Lpstar = %g' % (pp.on_slice, pp.Lpstar))
    # would want to replace by 
    # slice_fn(particle, Lpstar_min = np.NINF)
    # if it weren't for theta_unchanged=True which is hard to pass

    slice_sweep(pp, slice_fn, sigma = abs(slice_width), step_out = (slice_width > 0))
    #print(pp.pos)
    #print(pp.Lfn_ff)
    return (pp.pos, pp.ff, pp.Lfn_ff)
    
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
    K = pp.Kfn(theta); # TODO for v1, cannot use compact representation
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
	# Matlab chol(K) = upper Cholesky = lower Cholesky transpose = np.linalg.cholesky(K).T
    # pp.iK = inv(K)
    pp.U_invR = K.inv_add_diag_cholesky_T(1/pp.aux_var) # Matlab chol(plus_diag(pp.iK, 1./pp.aux_var));
	# %pp.U_noise = chol(plus_diag(K, aux_var_vec));

def eval_particle_aux_surr(pp, Lpstar_min, Lfn, theta_Lprior, theta_unchanged = False):

    # Prior on theta
    Ltprior = theta_Lprior(pp.pos)
    if Ltprior == np.NINF:
        pp.on_slice = False
        pp.Lpstar = np.NINF
        return
    if not theta_unchanged:
        theta_changed(pp)

    # Update f|gg,nu,theta
    pp.ff = pp.U_invR.solve(pp.nu) + pp.U_invR.solve(pp.Sinv_g) # pp.U_invR\pp.nu + solve_chol(pp.U_invR, pp.Sinv_g);
	# solve_chol really means "solve knowing that matrix is triangular", but doesn't affect the matrices

    # % Compute joint probability and slice acceptability.
    # % I have dropped the constant: -0.5*length(pp.gg)*log(2*pi)
    # %Lgprior = -0.5*(pp.gg'*solve_chol(pp.U_noise, pp.gg)) - sum(log(diag(pp.U_noise)));
    # %pp.Lpstar = Ltprior + Lgprior + Lfn(pp.ff);
    # %
    # % This version doesn't need U_noise, but commenting out the U_noise line and
    # % using this version doesn't actually seem to be faster?
    Lfprior = -0.5 * pp.ff.T.dot(pp.U.solve(pp.ff)) - pp.U.diag_log_sum() #-0.5*(pp.ff'*solve_chol(pp.U, pp.ff)) - sum(log(diag(pp.U)));
    LJacobian = - pp.U_invR.diag_log_sum() # -sum(log(diag(pp.U_invR)));
#    %LJacobian = sum(log(diag(pp.U_R)));
    Lg_f = -0.5 * np.sum((pp.gg - pp.ff) ** 2) / pp.aux_var - np.sum(np.log(pp.aux_var * np.ones(pp.ff.shape[0]))) #-0.5*sum((pp.gg - pp.ff).^2)./pp.aux_var - sum(log(pp.aux_var.*ones(size(pp.ff))));
    pp.Lfn_ff = Lfn(pp.ff)
    pp.Lpstar = Ltprior + Lg_f + Lfprior + pp.Lfn_ff + LJacobian
    # log p(theta) + log p(g|f) + log p(f|theta) + log p(x|f) + log p()
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    if False:
        print('particle on slice? %g, pp.Lpstar = %g, Lpstar_min = %g' % (pp.on_slice, pp.Lpstar, Lpstar_min))