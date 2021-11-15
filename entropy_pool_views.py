import numpy as np


def get_F(tX):
    """

    :param tX: J*P matrix, J = number of simulation(scenarios), P = number of risk factors(prices)
    :return: F,M,f: F is a M*J matrix, M = number of INEQUALITY views, each row of F stands for a view.
                    F is used in lagrange optimizer, with Fp <= f
    """
    tF = np.array([
        tX[:, 0] - tX[:, 1],  # view 0: X0 <= X1
        tX[:, 1] - tX[:, 2],  # view 1: X1 <= X2
    ])
    tM, _ = tF.shape
    tf = np.zeros(tM)
    return tF, tM, tf


def get_views_relative_ranking(tX: np.ndarray, tMu: np.ndarray):
    """

    :param tX: J*P matrix, J = number of simulation(scenarios), P = number of risk factors(prices), with P>=2
    :param tMu: P*1 matrix(vector), from this vector get relative ranking order between columns of tX
    :return: F,M,f: F is a M*J matrix, M = P-1 = number of INEQUALITY views, each row of F stands for a view.
                    F is used in lagrange optimizer, with Fp <= f
    """

    x_order = tMu.argsort()  # ascending arguments order
    diff_list = []
    for i, j in zip(x_order[:-1:], x_order[1::]):
        diff_list.append(tX[:, i] - tX[:, j])
    tF = np.array(diff_list)
    tM, _ = tF.shape
    tf = -np.diff(np.sort(tMu))
    return tF, tM, tf


def get_views_ep_score_ranking(tX: np.ndarray, tEPScore: np.ndarray, t_ep_direction: str):
    """

    :param tX: J*P matrix, J = number of simulation(scenarios), P = number of risk factors(prices), with P>=2
    :param tEPScore: P*1 matrix(vector), from this vector get relative ranking order between columns of tX
                     each elements of this vector stands for the ep score of an assets
    :param t_ep_direction: "TF" :the lesser ep = the greater pe -> the lesser return
                        "MR" :the lesser ep = the greater pe -> the greater return
    :return: F,M,f: F is a M*J matrix, M = P-1 = number of INEQUALITY views, each row of F stands for a view.
                    F is used in lagrange optimizer, with Fp <= f
    """
    x_order = tEPScore.argsort()  # ascending arguments order
    diff_list = []
    for i, j in zip(x_order[:-1:], x_order[1::]):
        if t_ep_direction in ["TF", "tf"]:
            diff_list.append(tX[:, i] - tX[:, j])
        else:
            diff_list.append(tX[:, j] - tX[:, i])
    tF = np.array(diff_list)
    tM, _ = tF.shape
    tf = np.zeros(tM)
    return tF, tM, tf


def get_H(tX):
    """

    :param tX: J*P matrix, J = number of simulation(scenarios), P = number of risk factors(prices)
    :return: H,N,h: H is N*J matrix, H = number of EQUALITY views, each row of H stands for a view.
                    H is used in lagrange optimizer, with Hp = h
    """
    tJ, tP = tX.shape
    tH = np.array([
        np.ones(tJ)
    ])
    tN, _ = tH.shape
    th = np.ones(tN)
    return tH, tN, th
