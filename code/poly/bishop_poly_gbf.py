# BISHOP 1996: Regression w/ input-dep noise, a Bayesian treatment
# Using linear model with polynomial variance noise data, and Gaussian basis functions
# to model noise


#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from gb_bayes.utils.desmat_gaussian import gen_desmat_gaussian
from gb_bayes.utils.bayesHetReg import BayesianHetRegression

l_path = '/home/fineline/projects/het_regression/dataMisc'

#%%
# data model: y = 0.5 + 1.5 * x
# variance model: sigma^2 = 10 + 5 * x + 3.2 * x^2
def data_model(x):
  return 0.5 + 1.5 * x

def variance_model(x):
  return 10. + 5. * x + 3.2 * x**2

#%% load data
# data model: y = 0.5 + 1.5 * x
# variance model: sigma^2 = 10 + 5 * x + 3.2 * x^2
(x_min, x_max) = (-20, 20) # indep var range
N = 200 # num data points
input = np.loadtxt(os.path.join(l_path, 'poly_xt_vals_200.csv'), delimiter=',')
X, t = input[0], input[1]

## for plotting ground truth functions ################################################
N_cont = 200
Xcont = np.linspace(x_min, x_max, N_cont)
tcont = data_model(Xcont)

# ground truth for input-dep variance
X_var_cont = variance_model(Xcont)
# ground truth for input-dep stddev = sqrt(variance)
X_stddev_cont = np.sqrt(X_var_cont)
# ground truth for inverse variance, beta
X_invvar_cont = 1. / X_var_cont
# ground truth for log inverse variance, log(beta) = u^T.phi
X_loginvvar_cont = np.log(X_invvar_cont)

# s_path = '/home/fineline/projects/het_regression/code/poly'
# np.savetxt(os.path.join(s_path, 'loginvvar.csv'),
#            np.vstack((Xcont, X_loginvvar_cont)).T,
#            delimiter=',')
#######################################################################################

#%% plot data points, gt y(x) and loginvvar, and gt stddev error bars
fig, ax = plt.subplots(2, figsize=(8, 6))
ax[0].plot(X, t,'.', label='training data')
ax[0].plot(Xcont, tcont, label='ground truth')
# ax.plot(Xcont, X_var_cont, label='variance gt')
ax[0].fill_between(Xcont, tcont + X_stddev_cont, tcont - X_stddev_cont,
                alpha=0.2, label='gt (+- $\sigma$)')
ax[0].set(xlabel=r'$x$', ylabel=r'$t$', title='Training data and ground truths')
ax[0].legend()
ax[0].grid()

ax[1].plot(Xcont, X_loginvvar_cont, label=r'log($\beta$) = log($\sigma^{-2}$) gt')
ax[1].set(title='log inverse variance')
ax[1].grid()
fig.show()

#%% setup basis function params
# number of basis function for y(x), log inv var
M_w, M_u = 1, 17

mus_u = np.linspace(-20, 20, M_u)

# Best value from cross-validation
s_u = 4.5

#%% generate design matrices Phi and Psi
# add constant for bias term
Phi = np.vstack((np.ones(N), X)).T
# gbfs for loginvvar
Psi = gen_desmat_gaussian(X, params={'s': s_u, 'mus': mus_u})

#%%
# init w and u weights

## weight estimates from 'bishop_quad_gbf_wu_est.py' ##################################
# 17 u gbfs at [-20, -17.5, ... , 17.5, 20]
# u means = [-4.38801643 -1.48682671 -1.08303236 -0.70953241 -0.82788598 -0.52814552
#  -0.12702582 -0.35124571  0.92912775  1.44125187 -0.01656875 -0.25044227
#  -0.4027459  -0.63613996 -0.85924866 -0.7419815  -0.98968709 -1.82957644]

# start sort of close
u = np.array([-4.38801643, -1.48682671, -1.08303236, -0.70953241, -0.82788598,
              -0.52814552, -0.12702582, -0.35124571,  0.92912775,  1.44125187,
              -0.01656875, -0.25044227, -0.4027459,  -0.63613996, -0.85924866,
              -0.7419815,  -0.98968709, -1.82957644])
u_init = np.around(u, decimals=1)

# u_init = np.array([0.1, -0.3, 1.1, 1.1, -0.5])

est = BayesianHetRegression()

#%% use fit and predict methods
sgd_train_rate = 0.5  # if using sgd minimization

est.fit(Phi, Psi, t, method='nelder-mead', logging=True)
u_weights, w_weights = est.u_fit, est.w_fit

t_gbfs = np.vstack((np.ones(N_cont), Xcont)).T
loginvvar_gbfs = gen_desmat_gaussian(Xcont, params={'s': s_u, 'mus': mus_u})

t_gbfs_total, loginvvar_gbfs_total = est.predict(t_gbfs, loginvvar_gbfs,
                                                 noise_est=True)
var_gbfs_total = 1. / np.exp(loginvvar_gbfs_total)

#%% plot results and gbfs
def plot_result_with_gbf(t_vals, t_vals_plot, var_vals_plot, loginvvar_vals_plot,
                         w_means, u_means, loginvvar_gbfs,
                         t_gbfs_total, var_gbfs_total, loginvvar_gbfs_total):
  fig, ax = plt.subplots(figsize=(8, 6))

  # plot range
  xmin, xmax = -20.0, 20.0
  # ymin, ymax = -2.5, 2.5

  # training data
  ax.plot(X, t_vals, '.', label='train data')
  # t ground truth and prediction
  ax.plot(Xcont, t_vals_plot, label='t gt', color='chocolate')
  ax.plot(Xcont, t_gbfs_total, '--', label='t pred', color='chocolate')
  # log(invvar) gt and prediction
  ax.plot(Xcont, loginvvar_vals_plot, label=r'log($\beta$) gt', color='green')
  ax.plot(Xcont, loginvvar_gbfs_total, '--', label=r'log($\beta$) pred', color='green')
  # var gt and prediction
  # ax.plot(Xcont, var_vals_plot, label=r'$\sigma^2$ gt', color='saddlebrown')
  # ax.plot(Xcont, var_gbfs_total, '--', label=r'$\sigma^2$ pred', color='saddlebrown')

  # shaded t gt +/- 1 stddev gt
  ax.fill_between(Xcont, t_vals_plot + X_stddev_cont, t_vals_plot - X_stddev_cont,
                  alpha=0.2, label='t gt $\pm$ $\sigma$')

  # centers of u gbfs
  # ax.vlines(mus_u, ymin, ymax, linestyles=':', color='dodgerblue', label='u gbf cent')
  # u gbf bias factor
  ax.hlines(u_means[0], xmin, xmax, linestyles='-.', linewidth=1, label='u gbfs')
  # all the scaled loginvvar gbfs
  for i in range(1, loginvvar_gbfs.shape[1]):
    ax.plot(Xcont, u_means[i] * loginvvar_gbfs[:, i], '-.', linewidth=1)

  tit = str(len(w_means)-1) + ' w gbfs, ' + str(len(u_means)-1) + ' u gbfs'
  ax.set(title=tit, xlabel=r'$x$', ylabel=r'$t$', xlim=(xmin, xmax))
  ax.grid()
  ax.legend(ncol=2)
  fig.show()

plot_result_with_gbf(t, tcont, X_var_cont, X_loginvvar_cont,
                     w_weights, u_weights, loginvvar_gbfs,
                     t_gbfs_total, var_gbfs_total, loginvvar_gbfs_total)

#%% plot gt and predicted loginvvar
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(Xcont, X_loginvvar_cont, label=r'log($\beta$) gt')
ax.plot(Xcont, loginvvar_gbfs_total, label=r'log($\beta$) pred')
ax.set(title='log inverse variance')
ax.grid()
ax.legend()
fig.show()
