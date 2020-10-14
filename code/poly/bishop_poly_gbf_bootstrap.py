#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from gb_bayes.utils.desmat_gaussian import gen_desmat_gaussian
from gb_bayes.utils.bayesHetReg import BayesianHetRegression
from sklearn.utils import resample

l_path = '/home/fineline/projects/het_regression/dataMisc'

#%%
# data model: y = 0.5 + 1.5 * x
# variance model: sigma^2 = 10 + 5 * x + 3.2 * x^2
def data_model(x):
  return 0.5 + 1.5 * x

def variance_model(x):
  return 10. + 5. * x + 3.2 * x**2

#%% load data
N = 200  # num data points
data_in = np.loadtxt(os.path.join(l_path, 'poly_xt_vals_200.csv'), delimiter=',')
X, t = data_in[0], data_in[1]

#%% setup basis function params
# number of basis function for y(x), log inv var
M_w, M_u = 1, 17

mus_u = np.linspace(-20, 20, M_u)

# Best value from cross-validation
s_u = 4.5

#%% generate design matrices Phi and Psi
# Make Phi and Psi design matrices, for dataset X, gbf spacing s, gbf centers mus
def make_Phi_Psi(X, s, mus):
  Phi_mat = np.vstack((np.ones(len(X)), X)).T
  Psi_mat = gen_desmat_gaussian(X, params={'s': s, 'mus': mus})
  return Phi_mat, Psi_mat

#%% do bootstrap
num_boot_samp = 50
est = BayesianHetRegression()

# w, u weight fits from each bootstrap run
w_fit_boot_list, u_fit_boot_list = [], []

for i in range(num_boot_samp):
  print('bootstrap iteration %s / %s' % (i+1, num_boot_samp))
  # resample with replacement
  X_it, t_it = resample(X, t)
  # make design matrices
  Phi_it, Psi_it = make_Phi_Psi(X_it, s_u, mus_u)
  est.fit(Phi_it, Psi_it, t_it, method='nelder-mead', logging=False)
  # save fitted weights for this bootstrap iteration
  w_fit_boot_list.append(est.w_fit)
  u_fit_boot_list.append(est.u_fit)

#%% calc stddev of w_fit from bootstrap fits
w_fit_boot = np.array(w_fit_boot_list)

w0_fit_boot, w1_fit_boot = w_fit_boot[:, 0], w_fit_boot[:, 1]

w0_fit_av, w1_fit_av = np.mean(w0_fit_boot), np.mean(w1_fit_boot)
w0_fit_stddev = \
  np.sqrt((1 / (num_boot_samp - 1)) * np.sum((w0_fit_boot - w0_fit_av)**2))
w1_fit_stddev = \
  np.sqrt((1 / (num_boot_samp - 1)) * np.sum((w1_fit_boot - w1_fit_av)**2))

print('w0 = %s +- %s' % (w0_fit_av, w0_fit_stddev))
print('w1 = %s +- %s' % (w1_fit_av, w1_fit_stddev))

#%% continuous data for plotting fits
(x_min, x_max) = (-20, 20)  # indep var range
N_cont = 200
Xcont = np.linspace(x_min, x_max, N_cont)

Phicont, Psicont = make_Phi_Psi(Xcont, s_u, mus_u)

#%% create average function of all bootstrap fits for loginvvar
loginvvar_cont_boot = np.vstack([Psicont @ u_fit for u_fit in u_fit_boot_list])
loginvvar_cont_av = np.mean(loginvvar_cont_boot, axis=0)

#%%
# make ground truth loginvvar continuous data
X_var_cont = variance_model(Xcont)
X_invvar_cont = 1. / X_var_cont
X_loginvvar_cont = np.log(X_invvar_cont)

#%%
# plot gt and predicted average loginvvar
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(Xcont, X_loginvvar_cont, label=r'log($\beta$) gt')
ax.plot(Xcont, loginvvar_cont_av, label=r'log($\beta$) pred')
ax.set(title='log inverse variance')
ax.grid()
ax.legend()
fig.show()

#%%
# plot gt and all bootstrap predicted loginvvar
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(Xcont, X_loginvvar_cont, label=r'log($\beta$) gt')
for i in range(num_boot_samp):
  ax.plot(Xcont, loginvvar_cont_boot[i, :], linewidth=0.5)
ax.set(title='log inverse variance')
ax.grid()
ax.legend()
fig.show()
