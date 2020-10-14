# BISHOP 1996: Regression w/ input-dep noise, a Bayesian treatment
#
# Testing various functions. Using functions from https://github.com/amber-kshz/PRML
#
# 'BISHOP' refers to the paper, not the PRML book

#%%
import matplotlib.pyplot as plt
import numpy as np
from gb_bayes.utils.desmat_gaussian import gen_desmat_gaussian
from gb_bayes.utils.nb_3_funcs import BayesianRidgeRegression
np.random.seed(42)

#%%
# ground truth data model (from https://github.com/amber-kshz/PRML notebook 3)
def data_model(x):
  return np.sin(2*x) + 0.2*np.sin(x) + 0.1*x

#%% generate gt data

# 50 points, with x range from -3 to 3
N = 50
(x_min, x_max) = (-3, 3)
X = np.random.uniform(x_min, x_max, N)
ep = 0.3*np.random.randn(N)
t = data_model(X) + ep

N_cont = 200
Xcont = np.linspace(np.min(X), np.max(X), N_cont) # for plotting

plt.figure(figsize=(8,6))
plt.plot(X, t,'.', label='training data')
plt.plot(Xcont, data_model(Xcont), label='ground truth')
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.legend()
plt.grid(True)
plt.show()

#%% setup Gaussian basis function params
# 4 gbfs (as in BISHOP) with centers at (-3, -1, 1, 3)
M = 16
mus = np.linspace(x_min, x_max, M)
s = (x_max - x_min) / (M - 1) # spacing equal to distance between centers
# s = 0.5

#%%
def plot_result(pred_mean, pred_std):
  plt.figure(figsize=(8, 6))
  plt.plot(X, t, '.', label='training data')
  plt.plot(Xcont, pred_mean, '--', label='predictive mean')
  plt.plot(Xcont, data_model(Xcont), ':', label='ground truth')
  plt.fill_between(Xcont, pred_mean + pred_std, pred_mean - pred_std, alpha=0.2)
  plt.xlabel(r'$x$')
  plt.ylabel(r'$t$')
  plt.grid(True)
  plt.legend()
  plt.show()


def plot_prediction_fixed_hparams(s, mus, alpha, beta):
  # generating design matrix
  Phi = gen_desmat_gaussian(X, params={'s':s, 'mus':mus})
  Phi_test = gen_desmat_gaussian(Xcont, params={'s':s, 'mus':mus})

  est = BayesianRidgeRegression(alpha=alpha, beta=beta)
  est.fit(Phi, t, optimize_hyperparams=False)
  pred_mean, pred_std = est.predict(Phi_test, return_std=True)
  print(est.m)
  plot_result(pred_mean, pred_std)

plot_prediction_fixed_hparams(s, mus, alpha=1.0, beta=1.0)

#%%
Phi = gen_desmat_gaussian(X, params={'s':s, 'mus':mus})
Phi_test = gen_desmat_gaussian(Xcont, params={'s':s, 'mus':mus})
est = BayesianRidgeRegression(alpha=1.0, beta=1.0)
est.fit(Phi, t, optimize_hyperparams=False)
pred_mean, pred_std = est.predict(Phi_test, return_std=True)
w_means = est.m

#%%
# NEW
t_gbfs = gen_desmat_gaussian(Xcont, params={'s':s, 'mus':mus})
# weighted sum of gbfs. should equal predictive mean
t_gbfs_total = t_gbfs @ w_means

# plot the gbfs
def plot_result_with_gbf(pred_mean, pred_std):
  plt.figure(figsize=(8, 6))
  plt.plot(X, t, '.', label='training data')
  plt.plot(Xcont, pred_mean, '--', label='predictive mean')
  plt.plot(Xcont, data_model(Xcont), ':', label='ground truth')
  plt.fill_between(Xcont, pred_mean + pred_std, pred_mean - pred_std, alpha=0.2)

  # all the scaled gbfs
  for i in range(1, M):
    plt.plot(Xcont, w_means[i] * t_gbfs[:, i])

  # plt.plot(Xcont, t_gbfs_total, label='sum of gbfs') # should equal pred mean (it
  # does)
  plt.xlabel(r'$x$')
  plt.ylabel(r'$t$')
  plt.grid(True)
  plt.legend()
  plt.show()

plot_result_with_gbf(pred_mean, pred_std)
