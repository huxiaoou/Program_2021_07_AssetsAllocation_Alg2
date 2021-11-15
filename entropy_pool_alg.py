import numpy as np
from scipy.optimize import minimize
from skyrim.winterhold import timer


def entropy(ph: np.ndarray, p: np.ndarray) -> float:
    return (ph * (np.log(ph) - np.log(p))).sum()


def lagrange_l(ph: np.ndarray, t: np.ndarray, p: np.ndarray,
               F: np.ndarray, M: int, f: np.ndarray,
               H: np.ndarray, N: int, h: np.ndarray
               ) -> float:
    s0 = entropy(ph=ph, p=p)
    s1 = 0
    if M > 0:
        lbd = t[0:M]
        s1 = lbd.dot(F.dot(ph) - f)
    s2 = 0
    if N > 0:
        nu = t[M:(M + N)]
        s2 = nu.dot(H.dot(ph) - h)
    return s0 + s1 + s2


def ph_lbd_nu(t: np.ndarray, p: np.ndarray,
              F: np.ndarray, M: int,
              H: np.ndarray, N: int
              ) -> np.ndarray:
    e = np.ones(shape=p.shape)
    sum_F = 0
    if M > 0:
        lbd = t[0:M]
        sum_F = F.T.dot(lbd)
    sum_H = 0
    if N > 0:
        nu = t[M:(M + N)]
        sum_H = H.T.dot(nu)
    return np.exp(np.log(p) - e - sum_F - sum_H)


def ph_jac(t: np.ndarray, p: np.ndarray,
           F: np.ndarray, M: int,
           H: np.ndarray, N: int
           ):
    """
    :return: 0: partial derivatives of each element of ph about lambda, which has a shape of M * J
             each column stands for a element of ph
             1: partial derivatives of each element of ph about nu    , which has a shape of N * J
             each column stands for a element of ph
    """
    ph = ph_lbd_nu(t=t, p=p, F=F, M=M, H=H, N=N)
    ph_par_lbd = -F.dot(np.diag(ph)) if M > 0 else 0
    ph_par_nu = -H.dot(np.diag(ph)) if N > 0 else 0
    return ph_par_lbd, ph_par_nu


def lagrange_dual(t: np.ndarray, p: np.ndarray,
                  F: np.ndarray, M: int, f: np.ndarray,
                  H: np.ndarray, N: int, h: np.ndarray
                  ) -> float:
    ph = ph_lbd_nu(t=t, p=p, F=F, M=M, H=H, N=N)
    res = lagrange_l(ph=ph, t=t, p=p, F=F, M=M, f=f, H=H, N=N, h=h)
    return -res


# @timer
def optimize_entropy(p: np.ndarray,
                     F: np.ndarray, M: int, f: np.ndarray,
                     H: np.ndarray, N: int, h: np.ndarray,
                     verbose: bool
                     ) -> (np.ndarray, np.ndarray):
    opt_res = minimize(
        fun=lagrange_dual,
        x0=np.concatenate([np.ones(M), np.zeros(N)]),
        args=(p, F, M, f, H, N, h),
        bounds=[(0, None)] * M + [(None, None)] * N,
    )
    ph = ph_lbd_nu(t=opt_res.x, p=p, F=F, M=M, H=H, N=N)
    if not opt_res.success:
        if verbose:
            print("Failed in optimization for posterior distribution is.")
    # lbd = opt_res.x[0:M]
    # nu  = opt_res.x[M:M+N]
    return opt_res.x, ph


# --- weighted mean and covariance
def wgt_mu(X: np.ndarray, w: np.ndarray):
    """

    :param X: J * p matrix, J = number of observation, p = number of variable
    :param w: J * 1 matrix, weight for each observation
    :return: weighted average for each column of X
    """
    return X.T.dot(w)


def wgt_cov(X: np.ndarray, w: np.ndarray):
    """

    :param X: J * p matrix, J = number of observation, p = number of variable
    :param w: J * 1 matrix, weight for each observation
    :return: weighted covariance between columns of X
    """
    m = wgt_mu(X, w).reshape(-1, 1)
    s2 = X.T.dot(np.diag(w)).dot(X)
    return s2 - m.dot(m.T)
