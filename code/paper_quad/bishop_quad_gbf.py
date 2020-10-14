# BISHOP 1996: Regression w/ input-dep noise, a Bayesian treatment
#
# Testing various functions. Using functions from https://github.com/amber-kshz/PRML
#
# 'BISHOP' refers to the paper, not the PRML book

#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from gb_bayes.utils.desmat_gaussian import gen_desmat_gaussian
from gb_bayes.utils.bayesHetReg import BayesianHetRegression

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
LOAD_DATA, SAVE_DATA = True, False

# Either read in or generate ground truth data
if LOAD_DATA: # read in data
  data_in = np.loadtxt(os.path.join(l_path, 'sin_xt_vals_300.csv'), delimiter=',')
  X, t = data_in[:, 0], data_in[:, 1]
else: # gen new data
  np.random.seed(42)
  # training data indep var (x) values
  X = np.random.uniform(x_min, x_max, N)
  # noise from x-dep variance model
  ep = np.array([np.random.normal(0., np.sqrt(variance_model(el))) for el in X])
  # training data hypothesis values y(x)
  t = data_model(X) + ep
  if SAVE_DATA:
    np.savetxt(os.path.join(l_path, 'sin_xt_vals_300.csv'), np.vstack((X, t)).T,
               delimiter=',')

## for plotting ground truth functions ################################################
# ground truth hypothesis
# Xcont = np.linspace(np.min(X), np.max(X), N)
Xcont = np.linspace(x_min, x_max, N)
tcont = data_model(Xcont)

# ground truth for input-dep variance
X_var_cont = variance_model(Xcont)
# ground truth for input-dep stddev = sqrt(variance)
X_stddev_cont = np.sqrt(X_var_cont)
# ground truth for inverse variance, beta
X_invvar_cont = 1. / X_var_cont
# ground truth for log inverse variance, log(beta) = u^T.phi
X_loginvvar_cont = np.log(X_invvar_cont)
#######################################################################################


#%% plot data points, gt y(x) and loginvvar, and gt stddev error bars
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(X, t,'.', label='training data')
ax.plot(Xcont, tcont, label='ground truth')
ax.plot(Xcont, X_var_cont, label='variance gt')
ax.plot(Xcont, X_loginvvar_cont, label=r'log($\beta$) = log($\sigma^{-2}$) gt')
ax.fill_between(Xcont, tcont + X_stddev_cont, tcont - X_stddev_cont,
                alpha=0.2, label='gt (+- $\sigma$)')
ax.set(xlabel=r'$x$', ylabel=r'$t$', title='Training data and ground truths')
ax.legend()
ax.grid()
fig.show()

#%% setup Gaussian basis function params
# number of basis function for y(x), log inv var
M_w, M_u = 4, 4

# mus are centers of gbfs, s is spacing between gbfs
# 4 w gbfs, w/ edge gbfs at s_w/2 from boundary of data.
mus_w = np.array([-2.25, -0.75, 0.75, 2.25])
s_w = 1.5

# 4 u gbfs, w/ edge gbfs at s_u/2 from boundary of data.
mus_u = np.array([-2.25, -0.75, 0.75, 2.25])
s_u = 2.25


#%%
# generate y(x) gbf design matrix Phi from input data X
Phi = gen_desmat_gaussian(X, params={'s': s_w, 'mus': mus_w})
# generate log(beta) gbf design matrix Psi from input data X
Psi = gen_desmat_gaussian(X, params={'s': s_u, 'mus': mus_u})

#%% init estimator
est = BayesianHetRegression()

#%% use fit and predict methods
est.fit(Phi, Psi, t, method='sgd', logging=True)
u_weights, w_weights = est.u_fit, est.w_fit

t_gbfs = gen_desmat_gaussian(Xcont, params={'s': s_w, 'mus': mus_w})
loginvvar_gbfs = gen_desmat_gaussian(Xcont, params={'s': s_u, 'mus': mus_u})

t_gbfs_total, loginvvar_gbfs_total = est.predict(t_gbfs, loginvvar_gbfs,
                                                 noise_est=True)
var_gbfs_total = 1. / np.exp(loginvvar_gbfs_total)

#%% plot results and gbfs
def plot_result_with_gbf(t_vals, t_vals_plot, var_vals_plot, loginvvar_vals_plot,
                         w_means, u_means, t_gbfs, loginvvar_gbfs,
                         t_gbfs_total, var_gbfs_total, loginvvar_gbfs_total):
  fig, ax = plt.subplots(figsize=(8, 6))

  # plot range
  xmin, xmax = -3.0, 3.0
  ymin, ymax = -2.5, 2.5

  # training data
  ax.plot(X, t_vals, '.', label='train data')
  # t ground truth and prediction
  ax.plot(Xcont, t_vals_plot, label='t gt', color='chocolate')
  ax.plot(Xcont, t_gbfs_total, '--', label='t pred', color='chocolate')
  # log(invvar) gt and prediction
  ax.plot(Xcont, loginvvar_vals_plot, label=r'log($\beta$) gt', color='green')
  ax.plot(Xcont, loginvvar_gbfs_total, '--', label=r'log($\beta$) pred', color='green')
  # var gt and prediction
  ax.plot(Xcont, var_vals_plot, label=r'$\sigma^2$ gt', color='saddlebrown')
  ax.plot(Xcont, var_gbfs_total, '--', label=r'$\sigma^2$ pred', color='saddlebrown')

  # shaded t gt +/- 1 stddev gt
  ax.fill_between(Xcont, t_vals_plot + X_stddev_cont, t_vals_plot - X_stddev_cont,
                  alpha=0.2, label='t gt $\pm$ $\sigma$')

  # # t gbf bias factor
  # ax.hlines(w_means[0], xmin, xmax, linestyles=':', label='w gbfs', color='grey')
  # # all the scaled t gbfs
  # for i in range(1, t_gbfs.shape[1]):
  #   ax.plot(Xcont, w_means[i] * t_gbfs[:, i], ':')

  # centers of u gbfs
  ax.vlines(mus_u, ymin, ymax, linestyles=':', color='dodgerblue', label='u gbf cent')
  # u gbf bias factor
  ax.hlines(u_means[0], xmin, xmax, linestyles='-.', linewidth=1, label='u gbfs')
  # all the scaled loginvvar gbfs
  for i in range(1, loginvvar_gbfs.shape[1]):
    ax.plot(Xcont, u_means[i] * loginvvar_gbfs[:, i], '-.', linewidth=1)

  tit = str(len(w_means)-1) + ' w gbfs, ' + str(len(u_means)-1) + ' u gbfs'
  ax.set(title=tit, xlabel=r'$x$', ylabel=r'$t$', xlim=(xmin, xmax), ylim=(ymin, ymax))
  ax.grid()
  ax.legend(ncol=2)
  fig.show()

plot_result_with_gbf(t, tcont, X_var_cont, X_loginvvar_cont,
                     w_weights, u_weights, t_gbfs, loginvvar_gbfs,
                     t_gbfs_total, var_gbfs_total, loginvvar_gbfs_total)

#%% Generate evidence M(u) landscape, for M_u = 3

# constant bias, u weights vary over (-3, 3)
bias = 0.7419
u1r = np.linspace(-3., 3., 60)
u2r = np.linspace(-3., 3., 60)
u3r = np.linspace(-3., 3., 60)

ev_vals = np.zeros((60, 60, 60))

for i in range(60):
  print('i= ' + str(i) + ' starting')
  for j in range(60):
    for k in range(60):
      u_val = np.array([bias, u1r[i], u2r[j], u3r[k]])
      ev_vals[i, j, k] = est.calc_evidence(Phi, Psi, u_val, t)

#%% print data on evidence landscape minimum
print('min ev val:', np.min(ev_vals))
# index of minimum of ev_vals
mi = np.unravel_index(np.argmin(ev_vals, axis=None), ev_vals.shape)
print('u vals at discretized ev min', bias, u1r[mi[0]], u2r[mi[1]], u3r[mi[2]])
print('minimized u_mp', u_weights)

print('ev at discretized ev min',
      est.calc_evidence(Phi,Psi,np.array([bias, u1r[mi[0]],u2r[mi[1]], u3r[mi[2]]]),t)
      )

print('ev at u_mp', est.calc_evidence(Phi, Psi, u_weights, t))

u_weights_d = np.array([bias, u1r[mi[0]], u2r[mi[1]], u3r[mi[2]]])
w_weights_d = est.calc_wmp(Phi, Psi, u_weights_d, t)

#%% save evidence landscape
np.save(os.path.join(l_path, 'ev_data', 'ev_vals_b_07419.npy'), ev_vals)
