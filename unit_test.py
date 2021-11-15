from entropy_pool_alg_alt import *
from entropy_pool_views import *

J = 10000

prior_mu = np.array([1, -1, 2])
prior_sd = np.array([2, 3, 2.5])
prior_cor = np.array([
    [1, 0.3, -0.2],
    [0.3, 1, 0.1],
    [-0.2, 0.1, 1],
])
prior_cov = (prior_cor * prior_sd).T * prior_sd

# print("mu")
# print(prior_mu)
# print("sd")
# print(prior_sd)
# print("cor")
# print(prior_cor)
# print("cov")
# print(prior_cov)

np.random.seed(1985)
X = np.random.multivariate_normal(mean=prior_mu, cov=prior_cov, size=J)
p = np.ones(J) / J

# F, M, f = get_F(tX=X)
F, M, f = get_views_relative_ranking(tX=X, tMu=np.array([-0.5, -0.2, 1]))
H, N, h = get_H(tX=X)

t1, ph1 = optimize_entropy(p=p, F=F, M=M, f=f, H=H, N=N, h=h, verbose=True)
t2, ph2 = optimize_entropy_jac(p=p, F=F, M=M, f=f, H=H, N=N, h=h, verbose=True)

print("=" * 120)
print("adjusted mu  =\n{}".format(wgt_mu(X, ph1)))
print("adjusted cov =\n{}".format(wgt_cov(X, ph1)))
print("adjusted cov2 =\n{}".format(wgt_cov2(X, ph1)))

print("=" * 120)
print("adjusted mu  =\n{}".format(wgt_mu(X, ph2)))
print("adjusted cov =\n{}".format(wgt_cov(X, ph2)))
print("adjusted cov2 =\n{}".format(wgt_cov2(X, ph2)))
