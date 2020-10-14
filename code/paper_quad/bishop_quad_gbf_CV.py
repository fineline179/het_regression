# BISHOP 1996: Regression w/ input-dep noise, a Bayesian treatment
#
# 'BISHOP' refers to the paper, not the PRML book
#
# Do crossvalidation on noise gbf width s_u

#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from gb_bayes.utils.desmat_gaussian import gen_desmat_gaussian
from gb_bayes.utils.bayesHetReg import BayesianHetRegression
from sklearn.model_selection import KFold
np.random.seed(42)

l_path = '/home/fineline/projects/het_regression/dataMisc'

#%%
# ground truth data model (from BISHOP fig 2). Looks like just sin(x)
def data_model(x):
  return np.sin(x)

# variance data model (from BISHOP fig 2)
def variance_model(x):
  return 0.1 + 0.044 * x**2

#%% generate/load ground truth training data

N = 300 # num data points
(x_min, x_max) = (-3, 3) # indep var range

# Load data
data_in = np.loadtxt(os.path.join(l_path, 'sin_xt_vals_300.csv'), delimiter=',')
X, t = data_in[:, 0], data_in[:, 1]

#%% setup Gaussian basis function params
# number of basis function for y(x), log inv var
M_w, M_u = 4, 4

# mus are centers of gbfs, s is spacing between gbfs
# 4 w gbfs, w/ edge gbfs at s_w/2 from boundary of data.
mus_w = np.array([-2.25, -0.75, 0.75, 2.25])
s_w = 1.5

# 4 u gbfs, w/ edge gbfs at s_u/2 from boundary of data.
mus_u = np.array([-2.25, -0.75, 0.75, 2.25])
s_u = 1.5


#%% create k-fold split for cross-val
num_splits = 10
k_fold = KFold(n_splits=num_splits, shuffle=True)

X_tr_kf, X_ts_kf, t_tr_kf, t_ts_kf = [], [], [], []
for train_ind, test_ind in k_fold.split(X):
  X_tr_kf.append(X[train_ind])
  X_ts_kf.append(X[test_ind])
  t_tr_kf.append(t[train_ind])
  t_ts_kf.append(t[test_ind])

#%%
est = BayesianHetRegression()

#%% train on folds for current s_u
s_u_range = list(np.arange(0.5, 3.5, 0.5))

# Calc the neg log likelihood of the test data
# - loglike = 0.5 * (\beta_i * \sum_{i=1}^{N_test} ((y(x_i) - t_i)^2 - log beta_i))
def neg_log_like(t_pred, t_test, beta_pred):
  negloglikes = 0.5 * (beta_pred * (t_pred - t_test)**2 - np.log(beta_pred))
  return np.sum(negloglikes)

def crossval_test_nll(s_u):
  nlls = []
  for i in range(num_splits):
    # make design matrices for training data for this fold
    Phi_tr = gen_desmat_gaussian(X_tr_kf[i], params={'s': s_w, 'mus': mus_w})
    Psi_tr = gen_desmat_gaussian(X_tr_kf[i], params={'s': s_u, 'mus': mus_u})

    # train on this fold
    print('minimizing s_u = %s, fold %s' % (s_u, i))
    est.fit(Phi_tr, Psi_tr, t_tr_kf[i], method='nelder-mead', logging=False)

    # make design matrices for test data for this fold
    Phi_ts = gen_desmat_gaussian(X_ts_kf[i], params={'s': s_w, 'mus': mus_w})
    Psi_ts = gen_desmat_gaussian(X_ts_kf[i], params={'s': s_u, 'mus': mus_u})

    # Predict at test points
    t_pred, loginvvar_pred = est.predict(Phi_ts, Psi_ts, noise_est=True)
    # inverse variance beta at test points
    beta_pred = np.exp(loginvvar_pred)

    # neg log-likelihood of test data for this fold
    nlls.append(neg_log_like(t_pred, t_ts_kf[i], beta_pred))

  nll_av = np.mean(nlls)
  print('nll_av = %s\n' % nll_av)
  return nll_av

nll_avs = [crossval_test_nll(s_u) for s_u in s_u_range]


#%%
fig, ax = plt.subplots()
ax.plot(s_u_range, nll_avs, '.')
fig.show()








