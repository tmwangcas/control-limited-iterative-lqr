import numpy as np


def boxQP(H, g, lower, upper, x0):
    """
    Minimize 0.5*x'*H*x + x'*g  s.t. lower<=x<=upper

     inputs:
        H            - positive definite matrix   (n * n)
        g            - bias vector                (n)
        lower        - lower bounds               (n)
        upper        - upper bounds               (n)

      optional inputs:
        x0           - initial state              (n)

     outputs:
        x            - solution                   (n)
        result       - result type (roughly, higher is better, see below)
        Hfree        - subspace cholesky factor   (n_free * n_free)
        free         - set of free dimensions     (n)
    """

    n        = H.shape[0]
    clamped  = np.zeros(n, dtype=bool)  # 1d array
    free     = np.ones(n, dtype=bool)
    oldvalue = 0
    result   = 0
    gnorm    = 0
    nfactor  = 0
    Hfree    = np.zeros((n,n))  # 2d array

    # initial state
    if x0 is not None and x0.shape[0] == n:
        x = np.clip(x0, lower, upper)
    else:
        LU = np.concatenate((lower[:,None], upper[:,None]), axis=1)
        # LU[np.isinf(LU)] = np.nan
        x = np.nanmean(LU, axis=1)

    x[np.isinf(x)] = 0

    # options
    maxIter         = 100       # maximum number of iterations
    minGrad         = 1e-8      # minimum norm of non-fixed gradient
    minRelImprove   = 1e-8      # minimum relative improvement
    stepDec         = 0.6       # factor for decreasing stepsize
    minStep         = 1e-22     # minimal stepsize for linesearch
    Armijo          = 0.1       # Armijo parameter (fraction of linear improvement required)
    verbosity       = 0         # verbosity

    # initial objective value
    value = np.dot(x, g) + 0.5*np.dot(np.dot(x, H), x)

    if verbosity > 0:
        print('==========\nStarting box-QP, dimension %-3d, initial value: %-12.3f\n' % (n, value))

    # main loop
    for i in range(maxIter):
        iter = i + 1

        if result != 0:
            break

        # check relative improvement
        if iter > 1 and (oldvalue - value) < minRelImprove * np.abs(oldvalue):
            result = 4
            break

        oldvalue = value

        # get gradient
        grad = g + np.dot(H, x)

        # find clamped dimensions
        old_clamped                        = clamped
        clamped                            = np.zeros(n, dtype=bool)  # 1d array
        clamped[(x == lower) & (grad > 0)] = True
        clamped[(x == upper) & (grad < 0)] = True
        free                               = ~clamped  # 1d array, dim = [free]

        # check for all clamped
        if np.all(clamped):
            result = 6
            break

        # factorize if clamped has changed
        if iter == 1:
            factorize = True
        else:
            factorize = np.any(old_clamped != clamped)

        if factorize:
            try:
                Hfree = np.linalg.cholesky(H[free,:][:,free])  # 2d array, dim = [free, free]
            except np.linalg.LinAlgError:
                result = -1
                break
            nfactor += 1

        # check gradient norm
        gnorm = np.linalg.norm(grad[free])  # 1d array, dim = [free]
        if gnorm < minGrad:
            result = 5
            break

        # get search direction
        grad_clamped = g + np.dot(H, (x*clamped))  # dim = [m]
        search       = np.zeros(n)  # dim = [m]
        search[free] = np.linalg.solve(-Hfree, np.linalg.solve(Hfree.T, grad_clamped[free])) - x[free]

        # check for descent direction
        sdotg = np.sum(search*grad)  # scalar
        if sdotg >= 0:
            break

        # armijo linesearch
        step  = 1
        nstep = 0
        xc    = np.clip(x + step*search, lower, upper)  # dim = [m]
        vc    = np.dot(xc, g) + 0.5*np.dot(np.dot(xc, H), xc)  # scalar
        while (vc - oldvalue) / (step * sdotg) < Armijo:
            step  = step * stepDec
            nstep += 1
            xc    = np.clip(x + step * search, lower, upper)
            vc    = np.dot(xc, g) + 0.5*np.dot(np.dot(xc, H), xc)
        if step < minStep:
            result = 2
            break

        if verbosity > 1:
            print('iter %-3d  value % -9.5g |g| %-9.3g  reduction %-9.3g  linesearch %g^%-2d  n_clamped %d\n' % (
                iter, vc, gnorm, oldvalue-vc, stepDec, nstep, np.sum(clamped)))

        # accept candidate
        x     = xc  # 1d array, dim = [m]
        value = vc  # scalar

    if iter >= maxIter:
        result = 1

    results = ['Hessian is not positive definite',          # result = -1
               'No descent direction found',                # result = 0  SHOULD NOT OCCUR
               'Maximum main iterations exceeded',          # result = 1
               'Maximum line-search iterations exceeded',   # result = 2
               'No bounds, returning Newton point',         # result = 3
               'Improvement smaller than tolerance',        # result = 4
               'Gradient norm smaller than tolerance',      # result = 5
               'All dimensions are clamped']                # result = 6

    if verbosity > 0:
        print('RESULT: %s.\niterations %d  gradient %-12.6g final value %-12.6g  factorizations %d\n' % results[result+1], iter, gnorm, value, nfactor)

    return x, result, Hfree, free

