from entropy_pool_alg import *

"""
created @ 2021-07-12
0.  alternative functions and methods to entropy pool algorithm
"""


def lagrange_dual_jac(t: np.ndarray, p: np.ndarray,
                      F: np.ndarray, M: int, f: np.ndarray,
                      H: np.ndarray, N: int, h: np.ndarray
                      ):
    ph = ph_lbd_nu(t=t, p=p, F=F, M=M, H=H, N=N)
    ph_par_lbd, ph_par_nu = ph_jac(t=t, p=p, F=F, M=M, H=H, N=N)
    J = len(p)
    e = np.ones(J)
    ans = np.zeros(shape=t.shape)
    if M > 0:
        lbd = t[0:M]
        g_par_lbd = ph_par_lbd.dot(np.log(ph) - np.log(p) + e) + F.dot(ph) - f + ph_par_lbd.dot(F.T.dot(lbd))
        if N > 0:
            nu = t[M:M + N]
            g_par_lbd += ph_par_lbd.dot(H.T.dot(nu))
        ans[0:M] = g_par_lbd
    if N > 0:
        nu = t[M:M + N]
        g_par_nu = ph_par_nu.dot(np.log(ph) - np.log(p) + e) + H.dot(ph) - h + ph_par_nu.dot(H.T.dot(nu))
        if M > 0:
            lbd = t[0:M]
            g_par_nu += ph_par_nu.dot(F.T.dot(lbd))
        ans[M:M + N] = g_par_nu
    return -ans


@timer
def optimize_entropy_jac(p: np.ndarray,
                         F: np.ndarray, M: int, f: np.ndarray,
                         H: np.ndarray, N: int, h: np.ndarray,
                         verbose: bool
                         ) -> (np.ndarray, np.ndarray):
    opt_res = minimize(
        fun=lagrange_dual,
        x0=np.concatenate([np.ones(M), np.zeros(N)]),
        args=(p, F, M, f, H, N, h),
        bounds=[(0, None)] * M + [(None, None)] * N,
        jac=lagrange_dual_jac,
        # method="SLSQP"
    )
    ph = ph_lbd_nu(t=opt_res.x, p=p, F=F, M=M, H=H, N=N)
    if not opt_res.success:
        if verbose:
            print("Failed in optimization for posterior distribution is.")
    # lbd = opt_res.x[0:M]
    # nu  = opt_res.x[M:M+N]
    return opt_res.x, ph


# --- weighted mean and covariance
def wgt_cov2(X: np.ndarray, w: np.ndarray):
    """

    :param X: J * p matrix, J = number of observation, p = number of variable
    :param w: J * 1 matrix, weight for each observation
    :return: weighted covariance between columns of X, this result should be the same as wgt cov
    """
    J, p = X.shape
    ans = np.zeros(shape=(p, p))
    for i in range(p):
        for k in range(p):
            ans[i, k] = (X[:, i] * X[:, k]).dot(w) - w.dot(X[:, i]) * w.dot(X[:, k])
    return ans
