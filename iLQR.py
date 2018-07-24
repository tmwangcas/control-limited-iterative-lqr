import numpy as np
import time
import warnings
from boxQP import boxQP


def iLQR(DYNCST, x0, u0, u_lims):
    """
    iLQR - solve the deterministic finite-horizon optimal control problem.

           minimize sum_i CST(x(:,i),u(:,i)) + CST(x(:,end))
               u
           s.t.  x(:,i+1) = DYN(x(:,i),u(:,i))

    Inputs
    ======
    DYNCST - A combined dynamics and cost function. It is called in
    three different formats.

     1) step:
      [xnew,c] = DYNCST(x,u,i) is called during the forward pass.
      Here the state x and control u are vectors: size(x)==[n 1],
      size(u)==[m 1]. The cost c and time index i are scalars.
      If Op.parallel==true (the default) then DYNCST(x,u,i) is be
      assumed to accept vectorized inputs: size(x,2)==size(u,2)==K

     2) final:
      [~,cnew] = DYNCST(x,nan) is called at the end the forward pass to compute
      the final cost. The nans indicate that no controls are applied.

     3) derivatives:
      [~,~,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = DYNCST(x,u,I) computes the
      derivatives along a trajectory. In this case size(x)==[n N+1] where N
      is the trajectory length. size(u)==[m N+1] with NaNs in the last column
      to indicate final-cost. The time indexes are I=(1:N).
      Dimensions match the variable names e.g. size(fxu)==[n n m N+1]
      note that the last temporal element N+1 is ignored for all tensors
      except cx and cxx, the final-cost derivatives.

    x0 - The initial state from which to solve the control problem.
    Should be a column vector. If a pre-rolled trajectory is available
    then size(x0)==[n N+1] can be provided and Op.cost set accordingly.

    u0 - The initial control sequence. A matrix of size(u0)==[m N]
    where m is the dimension of the control and N is the number of state
    transitions.

    u_lims - control limits

    Outputs
    =======
    x - the optimal state trajectory found by the algorithm.
        size(x)==[n N+1]

    u - the optimal open-loop control sequence.
        size(u)==[m N]

    L - the optimal closed loop control gains. These gains multiply the
        deviation of a simulated trajectory from the nominal trajectory x.
        size(L)==[m n N]

    Vx - the gradient of the cost-to-go. size(Vx)==[n N+1]

    Vxx - the Hessian of the cost-to-go. size(Vxx)==[n n N+1]

    cost - the costs along the trajectory. size(cost)==[1 N+1]
           the cost-to-go is V = fliplr(cumsum(fliplr(cost)))

    lambda - the final value of the regularization parameter

    trace - a trace of various convergence-related values. One row for each
            iteration, the columns of trace are
            [iter lambda alpha g_norm dcost z sum(cost) dlambda]
            see below for details.

    timing - timing information



    BIBTeX:

    @INPROCEEDINGS{
    author={Tassa, Y. and Mansard, N. and Todorov, E.},
    booktitle={Robotics and Automation (ICRA), 2014 IEEE International Conference on},
    title={Control-Limited Differential Dynamic Programming},
    year={2014}, month={May}, doi={10.1109/ICRA.2014.6907001}}
    """

    # ---------------------- user-adjustable parameters ------------------------
    Op = {'lims':           None,                        # control limits
          'parallel':       True,                        # use parallel line-search?
          # 'Alpha':          10**np.linspace(0,-3,11),    # backtracking coefficients
          'Alpha':          5*10**np.linspace(1,-3,21),  # backtracking coefficients
          'tolFun':         1e-7,                        # reduction exit criterion
          'tolGrad':        1e-4,                        # gradient exit criterion
          'maxIter':        1000,                        # maximum iterations
          'lambda':         1,                           # initial value for lambda
          'dlambda':        1,                           # initial value for dlambda
          'lambdaFactor':   1.6,                         # lambda scaling factor
          'lambdaMax':      1e10,                        # lambda maximum value
          'lambdaMin':      1e-6,                        # below this value lambda = 0
          'regType':        1,                           # regularization type 1: q_uu+lambda*eye(); 2: V_xx+lambda*eye()
          'zMin':           0,                           # minimal accepted reduction ratio
          'print':          2,                           # 0: no;  1: final; 2: iter; 3: iter, detailed
          'cost':           None}                        # initial cost for pre-rolled trajectory

    # --- initial sizes and controls
    n = x0.shape[0]  # dimension of state vector
    m = u0.shape[0]  # dimension of control vector
    N = u0.shape[1]  # number of state transitions
    u = u0           # initial control sequence, dim = [m, N]

    # --- proccess options
    Op['lims'] = u_lims
    verbosity  = Op['print']
    lamb       = Op['lambda']
    dlamb      = Op['dlambda']

    # --- initialize trace data structure
    trace = {'iter':            np.nan,
             'lambda':          np.nan,
             'dlambda':         np.nan,
             'cost':            np.nan,
             'alpha':           np.nan,
             'grad_norm':       np.nan,
             'improvement':     np.nan,
             'reduc_ratio':     np.nan,
             'time_derivs':     np.nan,
             'time_forward':    np.nan,
             'time_backward':   np.nan}
    trace = np.tile(trace, np.minimum(Op['maxIter'], int(1e6)))
    trace[0]['iter']    = 1
    trace[0]['lambda']  = lamb
    trace[0]['dlambda'] = dlamb

    # --- initial trajectory
    if x0.shape[1] == 1:
        diverge = True
        for alpha in Op['Alpha']:
            x, un, cost = forward_pass(x0, alpha*u, None, None, None, np.array([1]), DYNCST, Op['lims'])
            # simplistic divergence test
            if np.all(np.abs(x) < 1e8):
                x    = x.reshape(n,N+1)   # 2d array, dim = [n, N+1]
                u    = un.reshape(m,N)    # 2d array, dim = [m, N]
                cost = cost.reshape(N+1)  # 1d array, dim = [N+1]
                diverge = False
                break

    elif x0.shape[1] == N + 1:  # pre-rolled initial forward pass
        x = x0  # dim = [n, N+1]
        diverge = False
        if Op['cost'] is None:
            raise ValueError('pre-rolled initial trajectory requires cost')
        else:
            cost = Op['cost']

    else:
        raise ValueError('pre-rolled initial trajectory must be of correct length')

    trace[0]['cost'] = np.sum(cost)

    if diverge:
        Vx    = np.nan
        Vxx   = np.nan
        L     = np.zeros((N,m,n))
        cost  = None
        trace = trace[0]
        if verbosity > 0:
            print('\nEXIT: Initial control sequence caused divergence\n')
        return x, u, cost

    # constants, timers, counters
    flgChange  = 1
    dcost      = 0
    z          = 0
    expected   = 0
    print_head = 6  # print headings every print_head lines
    last_head  = print_head
    t_start    = time.time()
    diff_t     = np.zeros(Op['maxIter'])
    back_t     = np.zeros(Op['maxIter'])
    fwd_t      = np.zeros(Op['maxIter'])

    if verbosity > 0:
        print('\n=========== begin iLQG ===========\n')

    for i in range(Op['maxIter']):
        iter = i + 1
        trace[i]['iter'] = iter

        # ====== STEP 1: differentiate dynamics and cost along new trajectory
        if flgChange:
            t_diff = time.time()
            u_sup = np.concatenate((u, np.zeros((m,1))), axis=1)  # dim = [m, N+1]
            _, _, fx, fu, cx, cu, cxx, cuu, cux = DYNCST(x, u_sup)  # DYNCST(x,u), x dim = [n, N+1], u dim = [m, N+1]
            trace[i]['time_derivs'] = time.time() - t_diff
            flgChange = 0

        # ====== STEP 2: backward pass, compute optimal control law and cost-to-go
        backPassDone = 0

        while not backPassDone:
            t_back = time.time()
            diverge, Vx, Vxx, l, L, dV = back_pass(fx, fu, cx, cu, cxx, cuu, cux, lamb, Op['regType'], Op['lims'], u)
            # l dim   = [m, N]
            # L dim   = [N, m, n]
            # Vx dim  = [n, N+1]
            # Vxx dim = [N+1, n, n]
            # dV dim  = [2]

            trace[i]['time_backward']= time.time() - t_back

            if diverge:
                if verbosity > 2:
                    print('Cholesky failed at timestep %d.\n', diverge)
                dlamb = np.maximum(dlamb*Op['lambdaFactor'], Op['lambdaFactor'])
                lamb  = np.maximum(lamb*dlamb, Op['lambdaMin'])
                if lamb > Op['lambdaMax']:
                    break
                continue

            backPassDone = 1

        # check for termination due to small gradient
        g_norm = np.mean(np.amax(np.abs(l)/(np.abs(u) + 1), 0))  # scalar
        trace[i]['grad_norm'] = g_norm
        if g_norm < Op['tolGrad'] and lamb < 1e-5:
            dlamb = np.minimum(dlamb/Op['lambdaFactor'], 1/Op['lambdaFactor'])
            lamb  = lamb * dlamb * (lamb > Op['lambdaMin'])
            if verbosity > 0:
                print('\nSUCCESS: gradient norm < tolGrad\n')
            break

        # ====== STEP 3: line-search to find new control sequence, trajectory, cost
        fwdPassDone = 0

        if backPassDone:
            t_fwd = time.time()

            if Op['parallel']:  # parallel line-search
                xnew, unew, costnew = forward_pass(x0, u, L, x[:,0:N], l, Op['Alpha'], DYNCST, Op['lims'])
                # xnew dim = [K, n, N+1]
                # unew dim = [K, m, N]
                # cnew dim = [K, N+1]

                Dcost = np.sum(cost) - np.sum(costnew, 1)  # 1d array, dim = [K] (number of line search steps)
                dcost = np.amax(Dcost)  # scalar
                w     = np.argmax(Dcost)
                alpha = Op['Alpha'][w]
                expected = -alpha * (dV[0] + alpha * dV[1])  # scalar
                if expected > 0:
                    z = dcost / expected  # scalar
                else:
                    z = np.sign(dcost)
                    warnings.warn('non-positive expected reduction: should not occur')

                # if z > Op['zMin']:  # when z > 0
                #     fwdPassDone = 1
                #     xnew    = xnew[w,:,:]
                #     unew    = unew[w,:,:]
                #     costnew = costnew[w,:]

                # even if the cost does not descent, still take the step
                fwdPassDone = 1
                xnew    = xnew[w,:,:]
                unew    = unew[w,:,:]
                costnew = costnew[w,:]

            else:  # serial backtracking line-search
                for alpha in Op['Alpha']:
                    xnew, unew, costnew = forward_pass(x0, u + l*alpha, L, x[:,0:N], None, np.array([1]), DYNCST, Op['lims'])
                    xnew    = xnew.reshape(n,N+1)   # 2d array, dim = [n, N+1]
                    unew    = unew.reshape(m,N)     # 2d array, dim = [m, N]
                    costnew = costnew.reshape(N+1)  # 1d array, dim = [N+1]

                    dcost = np.sum(cost) - np.sum(costnew)
                    expected = -alpha * (dV[0] + alpha * dV[1])
                    if expected > 0:
                        z = dcost / expected
                    else:
                        z = np.sign(dcost)
                        warnings.warn('non-positive expected reduction: should not occur')
                    if z > Op['zMin']:
                        fwdPassDone = 1
                        break

            if not fwdPassDone:
                alpha = np.nan  # signals failure of forward pass

            trace[i]['time_forward'] = time.time() - t_fwd

        # ====== STEP 4: accept step (or not), draw graphics, print status
        # print headings
        if verbosity > 1 and last_head == print_head:
            last_head = 0
            print('%-12s%-12s%-12s%-12s%-12s%-12s' % ('iteration', 'cost', 'reduction', 'expected', 'gradient', 'log10(lambda)'))

        if fwdPassDone:
            # print status
            if verbosity > 1:
                print('%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f' % (iter, np.sum(cost), dcost, expected, g_norm, np.log10(lamb)))
                last_head += 1

            # decrease lambda
            dlamb = np.minimum(dlamb/Op['lambdaFactor'], 1/Op['lambdaFactor'])
            lamb  = lamb * dlamb * (lamb > Op['lambdaMin'])

            # accept changes
            x         = xnew
            u         = unew
            cost      = costnew
            flgChange = 1

            # terminate ?
            # if dcost < Op['tolFun']:
            #     if verbosity > 0:
            #         print('\nSUCCESS: cost change < tolFun\n')
            #     break

        else:  # no cost improvement
            # increase lambda
            dlamb = np.maximum(dlamb*Op['lambdaFactor'], Op['lambdaFactor'])
            lamb  = np.maximum(lamb*dlamb, Op['lambdaMin'])

            # print status
            if verbosity > 1:
                print('%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.1f' % (iter, 'NO STEP', dcost, expected, g_norm, np.log10(lamb)))
                last_head += 1

            # terminate ?
            if lamb > Op['lambdaMax']:
                if verbosity > 0:
                    print('\nEXIT: lambda > lambdaMax\n')
                break

        # update trace
        trace[i]['lambda']      = lamb
        trace[i]['dlambda']     = dlamb
        trace[i]['alpha']       = alpha
        trace[i]['improvement'] = dcost
        trace[i]['cost']        = np.sum(cost)
        trace[i]['reduc_ratio'] = z

        # calculate time
        diff_t[i] = trace[i]['time_derivs']
        back_t[i] = trace[i]['time_backward']
        fwd_t[i]  = trace[i]['time_forward']


    if iter == Op['maxIter']:
        if verbosity > 0:
            print('\nEXIT: Maximum iterations reached.\n')

    if iter is not None:
        diff_t  = np.sum(diff_t[~np.isnan(diff_t)])
        back_t  = np.sum(back_t[~np.isnan(back_t)])
        fwd_t   = np.sum(fwd_t[~np.isnan(fwd_t)])
        total_t = time.time() - t_start

        if verbosity > 0:
            print('\n'
                  'iterations:   %-3d\n' % iter,
                  'final cost:   %-12.7g\n' % np.sum(cost),
                  'final grad:   %-12.7g\n' % g_norm,
                  'final lambda: %-12.7e\n' % lamb,
                  'time / iter:  %-5.0f ms\n' % (1e3*total_t/iter),
                  'total time:   %-5.2f seconds, of which\n' % total_t,
                  '  derivs:     %-4.1f%%\n' % (diff_t*100/total_t),
                  '  back pass:  %-4.1f%%\n' % (back_t*100/total_t),
                  '  fwd pass:   %-4.1f%%\n' % (fwd_t*100/total_t),
                  '  other:      %-4.1f%% (graphics etc.)\n' % ((total_t-diff_t-back_t-fwd_t)*100/total_t),
                  '=========== end iLQG ===========\n')
    else:
        raise ValueError('Failure: no iterations completed, something is wrong.')

    return x, u, cost


def forward_pass(x0, u, L, x, du, Alpha, DYNCST, lims):
    # parallel forward-pass (rollout)
    # internally time is on the 3rd dimension,
    # to facillitate vectorized dynamics calls

    n  = x0.shape[0]
    K  = Alpha.shape[0]  # Alpha is 1d array, dim = [K]: number of line search in each step
    m  = u.shape[0]
    N  = u.shape[1]

    xnew        = np.zeros((N+1,n,K))
    xnew[0,:,:] = np.tile(x0, K)
    unew        = np.zeros((N,m,K))
    # cnew        = np.zeros((N+1,1,K))
    cnew        = np.zeros((N+1,K))

    for i in range(N):
        unew[i,:,:] = np.tile(u[:,i][:,None], K)  # unew[i,:,:] dim = [m, K]

        if du is not None:
            unew[i,:,:] += np.dot(du[:,i][:,None], Alpha[None,:])  # Alpha dim = [1, K]

        if L is not None:
            dx = xnew[i,:,:] - np.tile(x[:,i][:,None], K)  # dx dim = [n, K]
            unew[i,:,:] += np.dot(L[i,:,:], dx)  # L[i,:,:] dim = [m, n]

        if lims is not None:
            unew[i,:,:] = np.clip(unew[i,:,:], lims[:,0][:,None], lims[:,1][:,None])

        xnew[i+1,:,:], cnew[i,:], _, _, _, _, _, _, _ = DYNCST(xnew[i,:,:], unew[i,:,:])  # DYNCST(x,u), x dim = [n, K], u dim = [m, K]

    _, cnew[N,:], _, _, _, _, _, _, _ = DYNCST(xnew[N,:,:], np.zeros((m,K)))

    # put the time dimension in the columns
    xnew = np.transpose(xnew, (2,1,0))  # dim = [K, n, N+1]
    unew = np.transpose(unew, (2,1,0))  # dim = [K, m, N]
    # cnew = np.transpose(cnew, (2,1,0))  # dim = [K, 1, N+1]
    cnew = np.transpose(cnew, (1,0))  # dim = [K , N+1]

    return xnew, unew, cnew


def back_pass(fx, fu, cx, cu, cxx, cuu, cux, lamb, regType, lims, u):
    # Perform the Ricatti-Mayne backward pass

    # fx  dim = [N, n, n]
    # fu  dim = [N, n, m]
    # cx  dim = [n, N]
    # cu  dim = [m, N]
    # cxx dim = [N, n, n]
    # cuu dim = [N, m, m]
    # cux dim = [N, m, n]
    # this N includes the final step

    N = cx.shape[1]
    n = cx.shape[0]
    m = cu.shape[0]

    k   = np.zeros((m,N-1))
    K   = np.zeros((N-1,m,n))
    Vx  = np.zeros((n,N))
    Vxx = np.zeros((N,n,n))
    dV  = np.zeros(2)

    Vx[:,N-1]   = cx[:,N-1]
    Vxx[N-1:,:] = cxx[N-1,:,:]

    diverge = 0

    for i in reversed(range(N - 1)):
        Qu  = cu[:,i] + np.dot(fu[i,:,:].T, Vx[:,i+1])  # Qu dim = [m]
        Qx  = cx[:,i] + np.dot(fx[i,:,:].T, Vx[:,i+1])

        Qxx = cxx[i,:,:] + np.dot(np.dot(fx[i,:,:].T, Vxx[i+1,:,:]), fx[i,:,:])
        Quu = cuu[i,:,:] + np.dot(np.dot(fu[i,:,:].T, Vxx[i+1,:,:]), fu[i,:,:])
        Qux = cux[i,:,:] + np.dot(np.dot(fu[i,:,:].T, Vxx[i+1,:,:]), fx[i,:,:])

        Vxx_reg = Vxx[i+1,:,:] + lamb * np.eye(n) * (regType == 2)
        Qux_reg = cux[i,:,:] + np.dot(np.dot(fu[i,:,:].T, Vxx_reg), fx[i,:,:])
        QuuF    = cuu[i,:,:] + np.dot(np.dot(fu[i,:,:].T, Vxx_reg), fu[i,:,:]) + lamb * np.eye(m) * (regType == 1)  # QuuF dim = [m, m]

        if lims is None or lims[0,0] > lims[0,1]:
            # no control limits: Cholesky decomposition, check for non-PD
            try:
                R = np.linalg.cholesky(QuuF)
            except np.linalg.LinAlgError:
                diverge = i + 1
                return diverge, Vx, Vxx, k, K, dV

            # find control law
            kK  = np.linalg.solve(-R, np.linalg.solve(R.T, np.concatenate((Qu[:,None], Qux_reg), axis=1)))  # kK dim = [m, n+1]
            k_i = kK[:,0]      # k_i dim = [m]
            K_i = kK[:,1:n+1]  # K_i dim = [m, n]

        else:  # solve Quadratic Program
            lower = lims[:,0] - u[:,i]  # dim = [m]
            upper = lims[:,1] - u[:,i]

            k_i, result, R, free = boxQP(QuuF, Qu, lower, upper, k[:,np.minimum(i+1,N-2)])
            # QuuF dim = [m, m]
            # Qu dim = [m]
            # lower/upper dim = [m]
            # k[:,np.minimum(i+1,N-2)] dim = [m]

            if result < 1:
                diverge = i + 1
                return diverge, Vx, Vxx, k, K, dV

            K_i = np.zeros((m,n))
            if np.any(free):
                Lfree = np.linalg.solve(-R, np.linalg.solve(R.T, Qux_reg[free,:]))
                K_i[free,:] = Lfree

        # update cost-to-go approximation
        dV          = dV  + np.array([np.dot(k_i, Qu), 0.5*np.dot(np.dot(k_i, Quu), k_i)])
        Vx[:,i]     = Qx  + np.dot(np.dot(K_i.T, Quu), k_i) + np.dot(K_i.T, Qu)  + np.dot(Qux.T, k_i)
        Vxx[i,:,:]  = Qxx + np.dot(np.dot(K_i.T, Quu), K_i) + np.dot(K_i.T, Qux) + np.dot(Qux.T, K_i)
        Vxx[i,:,:]  = 0.5 * (Vxx[i,:,:] + Vxx[i,:,:].T)

        # save controls/gains
        k[:,i]   = k_i
        K[i,:,:] = K_i

    return diverge, Vx, Vxx, k, K, dV

