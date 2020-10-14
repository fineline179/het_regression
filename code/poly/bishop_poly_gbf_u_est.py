#%%
import matplotlib.pyplot as plt
import os
import numpy as np
from gb_bayes.utils.desmat_gaussian import gen_desmat_gaussian
from gb_bayes.utils.nb_3_funcs import BayesianRidgeRegression
np.random.seed(42)

#%% load/create gt data
l_path = '/home/fineline/projects/het_regression/code/poly'
invvargt = np.loadtxt(os.path.join(l_path,'loginvvar.csv'), delimiter=',')

N = invvargt.shape[0]
(x_min, x_max) = (-20, 20)
X = invvargt[:, 0]
t_nonoise = invvargt[:, 1]
loginvvar_nonoise = invvargt[:, 1]
# small uniform gaussian noise
loginvvar_ep = np.array([np.random.normal(0., 0.2) for el in X])

# data with gaussian noise added
loginvvar = loginvvar_nonoise + loginvvar_ep

# For plotting
Xcont = X
loginvvar_cont = loginvvar_nonoise

#%% plot ground truth loginvvar and inverse variance
plt.figure(figsize=(8,6))
plt.plot(X, loginvvar,'.', label='log inv var data with noise')
plt.plot(Xcont, loginvvar_cont, label=r'log($\beta$) = log($\sigma^{-2}$) gt')
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.legend()
plt.grid(True)
plt.show()

#%% setup basis function params
# number of basis function for y(x), log inv var
M_u = 17

# 7 gbf equally spaced from -15 to 15
mus_u = np.linspace(-20, 20, M_u)
s_u = 2.5

#%%
# gbfs for loginvvar
Psi = gen_desmat_gaussian(X, params={'s':s_u, 'mus':mus_u})

#%%
est_loginvvar = BayesianRidgeRegression(alpha=1.0, beta=2.0)
est_loginvvar.fit(Psi, loginvvar, optimize_hyperparams=False)

#%%
pred_mean_u, pred_std_u = est_loginvvar.predict(Psi, return_std=True)
u_means = est_loginvvar.m

print('u means =', u_means)

#%% plot the gbfs
def plot_result_with_gbf(target_vals, target_vals_plot, weight_means,
                         target_gbfs, target_gbf_total,
                         pred_mean, pred_std, plot_gbf_sum=False):
  plt.figure(figsize=(8, 6))
  # plt.plot(X, target_vals, '.', label='training data')
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
  plt.ylim(-8, 3)
  plt.grid(True)
  plt.legend()
  plt.show()

#%%
# y coords of loginvvar gbfs, for plotting
loginvvar_gbfs = gen_desmat_gaussian(Xcont, params={'s':s_u, 'mus':mus_u})
# weighted sum of gbfs. should equal predictive mean
loginvvar_gbfs_total = loginvvar_gbfs @ u_means

plot_result_with_gbf(loginvvar, loginvvar_cont, u_means,
                     loginvvar_gbfs, loginvvar_gbfs_total,
                     pred_mean_u, pred_std_u, plot_gbf_sum=False)
