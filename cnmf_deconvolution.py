import numpy as np
import scipy
from scipy import signal
from past.utils import old_div
from cnmf_oasis import oasisAR1, constrained_oasisAR1


def constrained_foopsi(
        fluor, bl=None,  c1=None, g=None,  sn=None, p=1,
        bas_nonneg=True, noise_range=[.25, .5],
        noise_method='logmexp', lags=5, fudge_factor=1., verbosity=False,
        solvers=None, optimize_g=0, s_min=None, **kwargs):
    """
    Infer the most likely discretized spike train underlying a fluorescence
    trace. It relies on a noise constrained deconvolution approach.
    Args:
        fluor: np.ndarray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.
        bl: [optional] float
            Fluorescence baseline value. If no value is given, then bl is
            estimated from the data.
        c1: [optional] float
            value of calcium at time 0
        g: [optional] list,float
            Parameters of the AR process that models the fluorescence impulse
            response. Estimated from the data if no value is given
        sn: float, optional
            Standard deviation of the noise distribution.  If no value is
            given, then sn is estimated from the data.
        p: int
            order of the autoregression model
        method_deconvolution: [optional] string
            solution method for basis projection pursuit 'cvx' or 'cvxpy' or
            'oasis'
        bas_nonneg: bool
            baseline strictly non-negative
        noise_range:  list of two elms
            frequency range for averaging noise PSD
        noise_method: string
            method of averaging noise PSD
        lags: int
            number of lags for estimating time constants
        fudge_factor: float
            fudge factor for reducing time constant bias
        verbosity: bool
             display optimization details
        solvers: list string
            primary and secondary (if problem unfeasible for approx solution)
            solvers to be used with cvxpy, default is ['ECOS','SCS']
        optimize_g : [optional] int, only applies to method 'oasis'
            Number of large, isolated events to consider for optimizing g.
            If optimize_g=0 (default) the provided or estimated g is not
            further optimized.
        s_min : float, optional, only applies to method 'oasis'
            Minimal non-zero activity within each bin (minimal 'spike size').
            For negative values the threshold is abs(s_min) * sn * sqrt(1-g)
            If None (default) the standard L1 penalty is used
            If 0 the threshold is determined automatically such that
            RSS <= sn^2 T
    Returns:
        c: np.ndarray float
            The inferred denoised fluorescence signal at each time-bin.
        bl, c1, g, sn : As explained above
        sp: ndarray of float
            Discretized deconvolved neural activity (spikes)
        lam: float
            Regularization parameter
    Raises:
        Exception("You must specify the value of p")
        Exception('OASIS is currently only implemented for p=1 and p=2')
        Exception('Undefined Deconvolution Method')
    References:
        * Pnevmatikakis et al. 2016. Neuron, in press,
        http://dx.doi.org/10.1016/j.neuron.2015.11.037
        * Machado et al. 2015. Cell 162(2):338-350
    \image: docs/img/deconvolution.png
    \image: docs/img/evaluationcomponent.png
    """

    if g is None or sn is None:
        # Estimate noise standard deviation and AR coefficients,
        # if they are not present
        g, sn = estimate_parameters(
            fluor, p=p, sn=sn, g=g, range_ff=noise_range,
            method=noise_method, lags=lags, fudge_factor=fudge_factor)
    lam = None

    penalty = 1 if s_min is None else 0
    if p == 1:
        if bl is None:
            # Infer the most likely discretized spike train underlying
            # an AR(1) fluorescence trace. Solves the noise constrained sparse
            # non-negative deconvolution problem min |s|_1 subject to:
            # |c-y|^2 = sn^2 T and s_t = c_t-g c_{t-1} >= 0
            c, sp, bl, g, lam = constrained_oasisAR1(
                fluor.astype(np.float32), g[0], sn, optimize_b=True,
                b_nonneg=bas_nonneg, optimize_g=optimize_g, penalty=penalty,
                s_min=0 if s_min is None else s_min)
        else:
            c, sp, _, g, lam = constrained_oasisAR1(
                (fluor - bl).astype(np.float32), g[0], sn, optimize_b=False,
                penalty=penalty)

        c1 = c[0]

        # remove intial calcium to align with the other foopsi methods
        # it is added back in function constrained_foopsi_parallel of
        # temporal.py
        c -= c1 * g**np.arange(len(fluor))
    elif p == 2:
        if bl is None:
            c, sp, bl, g, lam = constrained_oasisAR2(
                fluor.astype(np.float32), g, sn, optimize_b=True,
                b_nonneg=bas_nonneg, optimize_g=optimize_g,
                penalty=penalty)
        else:
            c, sp, _, g, lam = constrained_oasisAR2(
                (fluor - bl).astype(np.float32), g, sn, optimize_b=False,
                penalty=penalty)
        c1 = c[0]
        d = (g[0] + np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
        c -= c1 * d**np.arange(len(fluor))
    else:
        raise Exception(
            'OASIS is currently only implemented for p=1 and p=2')
    g = np.ravel(g)

    return c, bl, c1, g, sn, sp, lam


def _nnls(KK, Ky, s=None, mask=None, tol=1e-9, max_iter=None):
    """
    Solve non-negative least squares problem
    ``argmin_s || Ks - y ||_2`` for ``s>=0``
    Args:
        KK : array, shape (n, n)
            Dot-product of design matrix K transposed and K, K'K
        Ky : array, shape (n,)
            Dot-product of design matrix K transposed and target vector y, K'y
        s : None or array, shape (n,), optional, default None
            Initialization of deconvolved neural activity.
        mask : array of bool, shape (n,), optional, default (True,)*n
            Mask to restrict potential spike times considered.
        tol : float, optional, default 1e-9
            Tolerance parameter.
        max_iter : None or int, optional, default None
            Maximum number of iterations before termination.
            If None (default), it is set to len(KK).
    Returns:
        s : array, shape (n,)
            Discretized deconvolved neural activity (spikes)
    References:
        Lawson C and Hanson RJ, SIAM 1987
        Bro R and DeJong S, J Chemometrics 1997
    """

    if mask is None:
        mask = np.ones(len(KK), dtype=bool)
    else:
        KK = KK[mask][:, mask]
        Ky = Ky[mask]
    if s is None:
        s = np.zeros(len(KK))
        l = Ky.copy()
        P = np.zeros(len(KK), dtype=bool)
    else:
        s = s[mask]
        P = s > 0
        l = Ky - KK[:, P].dot(s[P])
    i = 0
    if max_iter is None:
        max_iter = len(KK)
    for i in range(max_iter):  # max(l) is checked at the end, should do at least one iteration
        w = np.argmax(l)
        P[w] = True

        try:  # likely unnnecessary try-except-clause for robustness sake
            mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
        except:
            mu = np.linalg.inv(KK[P][:, P] + tol * np.eye(P.sum())).dot(Ky[P])
            print(r'added $\epsilon$I to avoid singularity')
        while len(mu > 0) and min(mu) < 0:
            a = min(s[P][mu < 0] / (s[P][mu < 0] - mu[mu < 0]))
            s[P] += a * (mu - s[P])
            P[s <= tol] = False
            try:
                mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
            except:
                mu = np.linalg.inv(KK[P][:, P] + tol *
                                   np.eye(P.sum())).dot(Ky[P])
                print(r'added $\epsilon$I to avoid singularity')
        s[P] = mu.copy()
        l = Ky - KK[:, P].dot(s[P])
        if max(l) < tol:
            break
    tmp = np.zeros(len(mask))
    tmp[mask] = s
    return tmp


def onnls(y, g, lam=0, shift=100, window=None, mask=None, tol=1e-9,
          max_iter=None):
    """ Infer the most likely discretized spike train underlying an AR(2)
    fluorescence trace. Solves the sparse non-negative deconvolution problem
    ``argmin_s 1/2|Ks-y|^2 + lam |s|_1`` for ``s>=0``
    Args:
        y : array of float, shape (T,)
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.
        g : array, shape (p,)
            if p in (1,2):
                Parameter(s) of the AR(p) process that models the fluorescence
                impulse response.
            else:
                Kernel that models the fluorescence impulse response.
        lam : float, optional, default 0
            Sparsity penalty parameter lambda.

        shift : int, optional, default 100
            Number of frames by which to shift window from on run of NNLS
            to the next.

        window : int, optional, default None (200 or larger dependend on g)
            Window size.

        mask : array of bool, shape (n,), optional, default (True,)*n
            Mask to restrict potential spike times considered.

        tol : float, optional, default 1e-9
            Tolerance parameter.

        max_iter : None or int, optional, default None
            Maximum number of iterations before termination.
            If None (default), it is set to window size.
    Returns:
        c : array of float, shape (T,)
            The inferred denoised fluorescence signal at each time-bin.

        s : array of float, shape (T,)
            Discretized deconvolved neural activity (spikes).
    References:
        Friedrich J and Paninski L, NIPS 2016
        Bro R and DeJong S, J Chemometrics 1997
    """

    T = len(y)
    if mask is None:
        mask = np.ones(T, dtype=bool)
    if window is None:
        w = max(200, len(g) if len(g) > 2 else
                int(-5 / np.log(g[0] if len(g) == 1 else
                    (g[0] + np.sqrt(g[0] * g[0] + 4 * g[1])) / 2)))
    else:
        w = window
    w = min(T, w)
    K = np.zeros((w, w))

    if len(g) == 1:  # kernel for AR(1)
        _y = y - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        h = np.exp(np.log(g[0]) * np.arange(w))
        for i in range(w):
            K[i:, i] = h[:w - i]

    elif len(g) == 2:  # kernel for AR(2)
        _y = y - lam * (1 - g[0] - g[1])
        _y[-2] = y[-2] - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        d = (g[0] + np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
        r = (g[0] - np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
        if d == r:
            h = np.exp(np.log(d) * np.arange(1, w + 1)) * np.arange(1, w + 1)
        else:
            h = (np.exp(np.log(d) * np.arange(1, w + 1)) -
                 np.exp(np.log(r) * np.arange(1, w + 1))) / (d - r)
        for i in range(w):
            K[i:, i] = h[:w - i]

    else:  # arbitrary kernel
        h = g
        for i in range(w):
            K[i:, i] = h[:w - i]
        a = np.linalg.inv(K).sum(0)
        _y = y - lam * a[0]
        _y[-w:] = y[-w:] - lam * a

    s = np.zeros(T)
    KK = K.T.dot(K)
    for i in range(0, max(1, T - w), shift):
        s[i:i + w] = _nnls(KK, K.T.dot(_y[i:i + w]), s[i:i + w],
                           mask=mask[i:i + w], tol=tol, max_iter=max_iter)[:w]

        # subtract contribution of spikes already committed to
        _y[i:i + w] -= K[:, :shift].dot(s[i:i + shift])
    s[i + shift:] = _nnls(KK[-(T - i - shift):, -(T - i - shift):],
                          K[:T - i - shift, :T - i -
                              shift].T.dot(_y[i + shift:]),
                          s[i + shift:], mask=mask[i + shift:])
    c = np.zeros_like(s)
    for t in np.where(s > tol)[0]:
        c[t:t + w] += s[t] * h[:min(w, T - t)]
    return c, s


def constrained_oasisAR2(
        y, g, sn, optimize_b=True, b_nonneg=True, optimize_g=0, decimate=5,
        shift=100, window=None, tol=1e-9, max_iter=1, penalty=1):
    """ Infer the most likely discretized spike train underlying an AR(2)
    fluorescence trace. Solves the noise constrained sparse non-negative
    deconvolution problem min (s)_1 subject to (c-y)^2 = sn^2 T and
    s_t = c_t-g1 c_{t-1}-g2 c_{t-2} >= 0
    Args:
        y : array of float
            One dimensional array containing the fluorescence intensities
            (with baseline already subtracted) with one entry per time-bin.

        g : (float, float)
            Parameters of the AR(2) process that models the fluorescence
            impulse response.

        sn : float
            Standard deviation of the noise distribution.

        optimize_b : bool, optional, default True
            Optimize baseline if True else it is set to 0, see y.

        b_nonneg: bool, optional, default True
            Enforce strictly non-negative baseline if True.

        optimize_g : int, optional, default 0
            Number of large, isolated events to consider for optimizing g.
            No optimization if optimize_g=0.

        decimate : int, optional, default 5
            Decimation factor for estimating hyper-parameters faster on
            decimated data.

        shift : int, optional, default 100
            Number of frames by which to shift window from on run of NNLS to
            the next.

        window : int, optional, default None (200 or larger dependend on g)
            Window size.

        tol : float, optional, default 1e-9
            Tolerance parameter.

        max_iter : int, optional, default 1
            Maximal number of iterations.

        penalty : int, optional, default 1
            Sparsity penalty. 1: min (s)_1  0: min (s)_0
    Returns:
        c : array of float
            The inferred denoised fluorescence signal at each time-bin.

        s : array of float
            Discretized deconvolved neural activity (spikes).

        b : float
            Fluorescence baseline value.

        (g1, g2) : tuple of float
            Parameters of the AR(2) process that models the fluorescence
            impulse response.

        lam : float
            Sparsity penalty parameter lambda of dual problem.
    References:
        Friedrich J and Paninski L, NIPS 2016
        Friedrich J, Zhou P, and Paninski L, arXiv 2016
    """

    T = len(y)
    d = (g[0] + np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
    r = (g[0] - np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
    if window is None:
        window = int(min(T, max(200, -5 / np.log(d))))

    if not optimize_g:
        g11 = (np.exp(np.log(d) * np.arange(1, T + 1)) * np.arange(1, T + 1)) if d == r else \
            (np.exp(np.log(d) * np.arange(1, T + 1)) -
             np.exp(np.log(r) * np.arange(1, T + 1))) / (d - r)
        g12 = np.append(0, g[1] * g11[:-1])
        g11g11 = np.cumsum(g11 * g11)
        g11g12 = np.cumsum(g11 * g12)
        Sg11 = np.cumsum(g11)
        f_lam = 1 - g[0] - g[1]
    elif decimate == 0:  # need to run AR1 anyways for estimating AR coeffs
        decimate = 1
    thresh = sn * sn * T

    # get initial estimate of b and lam on downsampled data using AR1 model
    if decimate > 0:
        _, s, b, aa, lam = constrained_oasisAR1(
            y[:len(y) // decimate * decimate].reshape(-1, decimate).mean(1),
            d**decimate, sn / np.sqrt(decimate),
            optimize_b=optimize_b, b_nonneg=b_nonneg, optimize_g=optimize_g)
        if optimize_g:
            from scipy.optimize import minimize
            d = aa**(1. / decimate)
            if decimate > 1:
                s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
            r = estimate_time_constant(s, 1, fudge_factor=.98)[0]
            g[0] = d + r
            g[1] = -d * r
            g11 = (np.exp(np.log(d) * np.arange(1, T + 1)) -
                   np.exp(np.log(r) * np.arange(1, T + 1))) / (d - r)
            g12 = np.append(0, g[1] * g11[:-1])
            g11g11 = np.cumsum(g11 * g11)
            g11g12 = np.cumsum(g11 * g12)
            Sg11 = np.cumsum(g11)
            f_lam = 1 - g[0] - g[1]
        elif decimate > 1:
            s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
        lam *= (1 - d**decimate) / f_lam

        # this window size seems necessary and sufficient
        possible_spikes = [x + np.arange(-2, 3)
                           for x in np.where(s > s.max() / 10.)[0]]
        ff = np.array(possible_spikes, dtype=np.int).ravel()
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        mask = np.zeros(T, dtype=bool)
        mask[ff] = True
    else:
        b = np.percentile(y, 15) if optimize_b else 0
        lam = 2 * sn * np.linalg.norm(g11)
        mask = None
    if b_nonneg:
        b = max(b, 0)

    # run ONNLS
    c, s = onnls(y - b, g, lam=lam, mask=mask,
                 shift=shift, window=window, tol=tol)

    if not optimize_b:  # don't optimize b, just the dual variable lambda
        for _ in range(max_iter - 1):
            res = y - c
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4 * thresh:
                break

            # calc shift dlam, here attributed to sparsity penalty
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * \
                np.exp(np.log(d) * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f - 1

                # if and elif correct last 2 time points for |s|_1 instead |c|_1
                if i == len(ls) - 2:  # last pool
                    tmp[f] = (1. / f_lam if l == 0 else
                              (Sg11[l] + g[1] / f_lam * g11[l - 1]
                               + (g[0] + g[1]) / f_lam * g11[l]
                               - g11g12[l] * tmp[f - 1]) / g11g11[l])
                # secondlast pool if last one has length 1
                elif i == len(ls) - 3 and ls[-2] == T - 1:
                    tmp[f] = (Sg11[l] + g[1] / f_lam * g11[l]
                              - g11g12[l] * tmp[f - 1]) / g11g11[l]
                else:  # all other pools
                    tmp[f] = (Sg11[l] - g11g12[l] * tmp[f - 1]) / g11g11[l]
                l += 1
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]

            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + np.sqrt(bb * bb - aa * cc)) / aa
            except:
                db = -bb / aa

            # perform shift
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask,
                         shift=shift, window=window, tol=tol)
            db = np.mean(y - c) - b
            b += db
            lam -= db / f_lam

    else:  # optimize b
        db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g[0] - g[1])
        g_converged = False
        for _ in range(max_iter - 1):
            res = y - c - b
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4 * thresh:
                break
            # calc shift db, here attributed to baseline
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * \
                np.exp(np.log(d) * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f
                tmp[f] = (Sg11[l - 1] - g11g12[l - 1]
                          * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            tmp -= tmp.mean()
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + np.sqrt(bb * bb - aa * cc)) / aa
            except:
                db = -bb / aa

            # perform shift
            if b_nonneg:
                db = np.max(db, -b)
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask,
                         shift=shift, window=window, tol=tol)

            # update b and lam
            db = np.max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            lam -= db / f_lam

            # update g and b
            if optimize_g and (not g_converged):

                def getRSS(y, opt):
                    b, ld, lr = opt
                    if ld < lr:
                        return 1e3 * thresh
                    d, r = np.exp(ld), np.exp(lr)
                    g1, g2 = d + r, -d * r
                    tmp = b + onnls(y - b, [g1, g2], lam,
                                    mask=(s > 1e-2 * s.max()))[0] - y
                    return tmp.dot(tmp)

                result = minimize(
                    lambda x: getRSS(y, x), (b, np.log(d), np.log(r)),
                    bounds=((0 if b_nonneg else None, None),
                            (None, -1e-4), (None, -1e-3)),
                    method='L-BFGS-B',
                    options={'gtol': 1e-04, 'maxiter': 10, 'ftol': 1e-05})
                if abs(result['x'][1] - np.log(d)) < 1e-3:
                    g_converged = True
                b, ld, lr = result['x']
                d, r = np.exp(ld), np.exp(lr)
                g = (d + r, -d * r)
                c, s = onnls(y - b, g, lam=lam, mask=mask,
                             shift=shift, window=window, tol=tol)

                # update b and lam
                db = np.max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                lam -= db

    if penalty == 0:  # get (locally optimal) L0 solution
        def c4smin(y, s, s_min):
            ls = np.append(np.where(s > s_min)[0], T)
            tmp = np.zeros_like(s)
            l = ls[0]  # first pool
            tmp[:l] = np.max(0, np.exp(np.log(d) * np.arange(l)).dot(y[:l])
                             * (1 - d * d)
                             / (1 - d**(2 * l))) * np.exp(
                                np.log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f
                tmp[f] = (g11[:l].dot(y[f:f + l]) - g11g12[l - 1]
                          * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            return tmp

        spikesizes = np.sort(s[s > 1e-6])
        i = len(spikesizes) // 2
        l = 0
        u = len(spikesizes) - 1
        while u - l > 1:
            s_min = spikesizes[i]
            tmp = c4smin(y - b, s, s_min)
            res = y - b - tmp
            RSS = res.dot(res)
            if RSS < thresh or i == 0:
                l = i
                i = (l + u) // 2
                res0 = tmp
            else:
                u = i
                i = (l + u) // 2
        if i > 0:
            c = res0
            s = np.append([0, 0], c[2:] - g[0] * c[1:-1] - g[1] * c[:-2])

    return c, s, b, g, lam


def estimate_parameters(fluor, p=2, sn=None, g=None, range_ff=[0.25, 0.5],
                        method='logmexp', lags=5, fudge_factor=1.):
    """
    Estimate noise standard deviation and AR coefficients if they are not
    present.
    Args:
        p: positive integer
            order of AR system

        sn: float
            noise standard deviation, estimated if not provided.

        lags: positive integer
            number of additional lags where he autocovariance is computed

        range_ff : (1,2) array, nonnegative, max value <= 0.5
            range of frequency (x Nyquist rate) over which the spectrum is
            averaged

        method: string
            method of averaging: Mean, median, exponentiated mean of logvalues
            (default)

        fudge_factor: float (0< fudge_factor <= 1)
            shrinkage factor to reduce bias
    """

    if sn is None:
        sn = GetSn(fluor, range_ff, method)

    if g is None:
        if p == 0:
            g = np.array(0)
        else:
            g = estimate_time_constant(fluor, p, sn, lags, fudge_factor)

    return g, sn


def estimate_time_constant(fluor, p=2, sn=None, lags=5, fudge_factor=1.):
    """
    Estimate AR model parameters through the autocovariance function
    Args:
        fluor        : nparray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.

        p            : positive integer
            order of AR system

        sn           : float
            noise standard deviation, estimated if not provided.

        lags         : positive integer
            number of additional lags where he autocovariance is computed

        fudge_factor : float (0< fudge_factor <= 1)
            shrinkage factor to reduce bias
    Returns:
        g       : estimated coefficients of the AR process
    """

    if sn is None:
        sn = GetSn(fluor)

    lags += p
    xc = axcov(fluor, lags)
    xc = xc[:, np.newaxis]

    A = scipy.linalg.toeplitz(
        xc[lags + np.arange(lags)], xc[lags + np.arange(p)]
        ) - sn**2 * np.eye(lags, p)
    # g = np.linalg.lstsq(A, xc[lags + 1:])[0]
    g = np.linalg.lstsq(A, xc[lags + 1:], rcond=None)[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = old_div((gr + gr.conjugate()), 2.)
    np.random.seed(45)
    # We want some variability below, but it doesn't have to be random at
    # runtime. A static seed captures our intent, while still not disrupting
    # the desired identical results from runs.
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def GetSn(fluor, range_ff=[0.25, 0.5], method='logmexp'):
    """
    Estimate noise power through the power spectral density over the range of
    large frequencies
    Args:
        fluor    : nparray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.

        range_ff : (1,2) array, nonnegative, max value <= 0.5
            range of frequency (x Nyquist rate) over which the spectrum is
            averaged

        method   : string
            method of averaging: Mean, median, exponentiated mean of logvalues
            (default)
    Returns:
        sn       : noise standard deviation
    """

    ff, Pxx = signal.welch(fluor)  # scipy.signal.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(old_div(Pxx_ind, 2))),
        'median': lambda Pxx_ind: np.sqrt(np.median(old_div(Pxx_ind, 2))),
        'logmexp': lambda Pxx_ind: np.sqrt(
            np.exp(np.mean(np.log(old_div(Pxx_ind, 2)))))
    }[method](Pxx_ind)

    return sn


def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag
    Args:
        data : array
            Array containing fluorescence data

        maxlag : int
            Number of lags to use in autocovariance calculation
    Returns:
        axcov : array
            Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(old_div(xcov, T))


def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).
    Args:
        value : int
    Returns:
        exponent : int
    """

    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent
