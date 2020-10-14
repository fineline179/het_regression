# Estimate w and u params for BISHOP paper data.
# Data generated in 'bishop_quad_gbf.py'


#%%
import matplotlib.pyplot as plt
import os
import numpy as np
from gb_bayes.utils.desmat_gaussian import gen_desmat_gaussian
from gb_bayes.utils.nb_3_funcs import BayesianRidgeRegression
np.random.seed(42)

#%% load/create gt data
local_path = '/home/fineline/projects/het_regression/code/paper_quad'
ygt = np.loadtxt(os.path.join(local_path,'gt.csv'))
invvargt = np.loadtxt(os.path.join(local_path,'loginvvar.csv'))

N = ygt.shape[0]
(x_min, x_max) = (-3, 3)
X = ygt[:, 0]
t_nonoise = ygt[:, 1]
loginvvar_nonoise = invvargt[:, 1]
# small uniform gaussian noise
t_ep = np.array([np.random.normal(0., 0.1) for el in X])
loginvvar_ep = np.array([np.random.normal(0., 0.15) for el in X])

# data with gaussian noise added
t = t_nonoise + t_ep
loginvvar = loginvvar_nonoise + loginvvar_ep

# For plotting
Xcont = X
tcont = t_nonoise
loginvvar_cont = loginvvar_nonoise

#%% plot ground truth y(x) and inverse variance
plt.figure(figsize=(8,6))
plt.plot(X, t,'.', label='y data with noise')
plt.plot(Xcont, tcont, label='ground truth')
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.legend()
plt.grid(True)
plt.show()

#%% plot ground truth loginvvar and inverse variance
plt.figure(figsize=(8,6))
plt.plot(X, loginvvar,'.', label='log inv var data with noise')
plt.plot(Xcont, loginvvar_cont, label=r'log($\beta$) = log($\sigma^{-2}$) gt')
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.legend()
plt.grid(True)
plt.show()

#%% setup Gaussian basis function params
# 4 gbfs (as in BISHOP) with centers at (-3, -1, 1, 3)

# number of basis function for y(x), log inv var
M_w, M_u = 4, 3

# mus are centers of gbfs, s is spacing between gbfs
mus_w = np.linspace(x_min, x_max, M_w)
s_w = (x_max - x_min) / (M_w - 1)
# mus_u = np.linspace(x_min, x_max, M_u)
# s_u = (x_max - x_min) / (M_u - 1)

mus_u = np.array([-2, 0, 2])
s_u = 2

#%%
Phi = gen_desmat_gaussian(X, params={'s':s_w, 'mus':mus_w})
Psi = gen_desmat_gaussian(X, params={'s':s_u, 'mus':mus_u})

est_t = BayesianRidgeRegression(alpha=1.0, beta=1.0)
est_t.fit(Phi, t, optimize_hyperparams=False)
est_loginvvar = BayesianRidgeRegression(alpha=1.0, beta=1.0)
est_loginvvar.fit(Psi, loginvvar, optimize_hyperparams=False)

#%%
pred_mean_w, pred_std_w = est_t.predict(Phi, return_std=True)
w_means = est_t.m
pred_mean_u, pred_std_u = est_loginvvar.predict(Psi, return_std=True)
u_means = est_loginvvar.m

print('w means =', w_means)
print('u means =', u_means)

# TODO: u bias weight is smaller than average of data.. why?

#%% for plotting the gbfs
def plot_result_with_gbf(target_vals, target_vals_plot, weight_means,
                         target_gbfs, target_gbf_total,
                         pred_mean, pred_std, plot_gbf_sum=False):
  plt.figure(figsize=(8, 6))
  plt.plot(X, target_vals, '.', label='training data')
  plt.plot(Xcont, pred_mean, '--', label='predictive mean')
  plt.plot(Xcont, target_vals_plot, '-.', label='ground truth')
  plt.fill_between(Xcont, pred_mean + pred_std, pred_mean - pred_std, alpha=0.2)
  # bias factor
  plt.hlines(weight_means[0], x_min, x_max, label='bias weight')
  # all the scaled gbfs
  for i in range(1, target_gbfs.shape[1]):
    plt.plot(Xcont, weight_means[i] * target_gbfs[:, i])
  # sum of gbfs should equal pred mean (it does)
  if plot_gbf_sum:
    plt.plot(Xcont, target_gbf_total, '-.', label='sum of gbfs')
  plt.xlabel(r'$x$')
  plt.ylabel(r'$t$')
  plt.grid(True)
  plt.legend()
  plt.show()

#%%
# y coords of t gbfs, for plotting
t_gbfs = gen_desmat_gaussian(Xcont, params={'s':s_w, 'mus':mus_w})
# weighted sum of gbfs. should equal predictive mean
t_gbfs_total = t_gbfs @ w_means

plot_result_with_gbf(t, tcont, w_means,
                     t_gbfs, t_gbfs_total,
                     pred_mean_w, pred_std_w, plot_gbf_sum=True)

#%%
# y coords of loginvvar gbfs, for plotting
loginvvar_gbfs = gen_desmat_gaussian(Xcont, params={'s':s_u, 'mus':mus_u})
# weighted sum of gbfs. should equal predictive mean
loginvvar_gbfs_total = loginvvar_gbfs @ u_means

plot_result_with_gbf(loginvvar, loginvvar_cont, u_means,
                     loginvvar_gbfs, loginvvar_gbfs_total,
                     pred_mean_u, pred_std_u, plot_gbf_sum=True)
