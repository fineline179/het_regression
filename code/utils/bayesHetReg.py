# functions/classes for BISHOP paper regression problem
#
# Modeled on 'BayesianRidgeRegression' class from https://github.com/amber-kshz/PRML
# repo notebook 3

import math
import numpy as np
import scipy.optimize as opt

class BayesianHetRegression:

  def __init__(self, al_w=0.1, al_u=0.1):
    self.al_w = al_w
    self.al_u = al_u
    # fitted values of w and u weights
    self.w_fit = None
    self.u_fit = None

  def calc_wmp(self, Phi, Psi, u_mp, t):
    """
    Calculates w_mp, given u_mp

    Args:
      Phi ((N, M_w + 1) array): design matrix for y(x) gt
      Psi ((N, M_u + 1) array): design matrix for for log(beta) gt
      u_mp ((M_u + 1,) array): most probable current value of u weights
      t ((N,) array): target values

    Returns:
      w_mp ((M_w + 1,) array): most probable w vals for the given value of u_mp
    """

    N = np.shape(Phi)[0]
    M_w = np.shape(Phi)[1] - 1
    M_u = np.shape(Psi)[1] - 1

    B = np.diag(np.exp(Psi @ u_mp))

    S = np.linalg.inv(self.al_w * np.identity(M_w + 1) + Phi.T @ B @ Phi)
    w_mp = S @ Phi.T @ B @ t

    return w_mp

  def calc_evidence(self, Phi, Psi, u, t):
    """
    Calculates the evidence M(u)

    Args:
      Phi ((N, M_w + 1) array): design matrix for y(x) gt
      Psi ((N, M_u + 1) array): design matrix for for log(beta) gt
      u ((M_u + 1,) array): current value of u
      t ((N,) array): target values

    Returns:
      evidence (float): evidence for this value of u
    """

    N = np.shape(Phi)[0]
    M_w = np.shape(Phi)[1] - 1
    M_u = np.shape(Psi)[1] - 1

    # w_mp must be evaluated for each new value of u when evidence is calculated
    w_mp = self.calc_wmp(Phi, Psi, u, t)

    B = np.diag(np.exp(Psi @ u))

    A = self.al_w * np.identity(M_w + 1) + Phi.T @ B @ Phi

    evidence = 0.5 * (t - Phi @ w_mp).T @ B @ (t - Phi @ w_mp) + \
               0.5 * self.al_u * (u @ u) - 0.5 * np.sum(Psi @ u) + \
               0.5 * np.log(np.linalg.det(A))

    return evidence

  def calc_evidence_der(self, Phi, Psi, u, t):
    """
    Calculates the derivative of the evidence, dM(u) / du, wrt the the components of u

    Args:
      Phi ((N, M_w + 1) array): design matrix for y(x) gt
      Psi ((N, M_u + 1) array): design matrix for for log(beta) gt
      u ((M_u + 1,) array): current value of u
      t ((N,) array): target values

    Returns:
      evidence_der ((M_u + 1,) array): components of the derivative of M(u) wrt
       the (M_u + 1) components of u
    """

    N = np.shape(Phi)[0]
    M_w = np.shape(Phi)[1] - 1
    M_u = np.shape(Psi)[1] - 1

    # w_mp must be evaluated for each new value of u when evidence deriv is calculated
    w_mp = self.calc_wmp(Phi, Psi, u, t)

    B = np.diag(np.exp(Psi @ u))
    A = self.al_w * np.identity(M_w + 1) + Phi.T @ B @ Phi
    A_inv = np.linalg.inv(A)

    evidence_der = np.zeros(M_u + 1)

    for j in range(M_u + 1):
      # deriv of A wrt jth component of u
      dAduj = Phi.T @ np.diag(Psi[:, j]) @ B @ Phi
      evidence_der[j] = \
        0.5 * (t - Phi @ w_mp).T @ np.diag(Psi[:, j]) @ B @ (t - Phi @ w_mp) + \
        self.al_u * u[j] - 0.5 * np.sum(Psi[:, j]) + 0.5 * np.trace(A_inv @ dAduj)

    return evidence_der

  def fit(self, Phi, Psi, t, method='nelder-mead', **kwargs):
    """
    Fits weights w and u to training data in Phi, Psi, and t. Current methods include
    nelder-mead (no gradient required) and simple gradient descent.

    Args:
      Phi ((N, M_w + 1) array): training design matrix for y(x) gt
      Psi ((N, M_u + 1) array): training design matrix for for log(beta) gt
      t ((N,) array): training target values
      method (string): 'nelder-mead' or 'sgd'
      **kwargs: arguments to fitting function
    """

    if method == 'nelder-mead':
      self._fit_nelder_mead(Phi, Psi, t, **kwargs)
    elif method == 'sgd':
      self._fit_simple_grad_descent(Phi, Psi, t, **kwargs)

  def _fit_nelder_mead(self, Phi, Psi, t, logging=True, ev_scaling=1500.):
    if logging:
      print('fitting using nelder-mead\n')
    # curry BayesianHetRegression class functions to func of just u weights
    # (for feeding to scipy optimize functions)
    raw_calc_evidence = lambda u: self.calc_evidence(Phi, Psi, u, t) / ev_scaling

    M_u = np.shape(Psi)[1] - 1
    # initial value for u
    u_init = np.random.normal(0., 1., size=M_u + 1)
    res = opt.minimize(raw_calc_evidence, u_init, method='nelder-mead',
                       options={'maxiter': 20000})
    # set fitted values for w and u weights
    self.u_fit = res.x
    self.w_fit = self.calc_wmp(Phi, Psi, self.u_fit, t)

    if logging:
      print(str(res) + '\n\n' + 'w_mp = ' + str(self.w_fit) + '\nu_mp = ' +
            str(self.u_fit))

  def _fit_simple_grad_descent(self, Phi, Psi, t,
                               logging=True,
                               maxiter=5000, train_rate=0.05, log_freq=100,
                               ev_scaling=50.):
    if logging:
      print('fitting using simple gradient descent\n')
    M_u = np.shape(Psi)[1] - 1
    # initial value for u
    u = np.random.normal(0., 1., size=M_u + 1)

    for i in range(maxiter):
      ev_der = self.calc_evidence_der(Phi, Psi, u, t) / ev_scaling
      u -= train_rate * ev_der

      if logging and i % log_freq == 0:
        ev = self.calc_evidence(Phi, Psi, u, t) / ev_scaling
        it_string = 'iteration ' + str(i) + ': '
        print(it_string + 'evidence = ', ev)

    self.u_fit = u
    self.w_fit = self.calc_wmp(Phi, Psi, self.u_fit, t)

  def predict(self, Phi_test, Psi_test=None, noise_est=False):
    """
    Makes prediction on the input Phi_test (and Psi_test, if supplied).

    Args:
      Phi_test ((N_test, M_w + 1) array): test design matrix for y(x) gt
      Psi_test ((N_test, M_u + 1) array): test design matrix for for log(beta) gt
      noise_est (bool): whether to compute log(beta) for test data

    Returns:
      t_pred ((N_test,) array): y(x) gt predictions at test values
      loginvvar_pred ((N_test,) array): log(beta) predictions at test values
    """

    t_pred = Phi_test @ self.w_fit
    if not noise_est:
      return t_pred
    else:
      loginvvar_pred = Psi_test @ self.u_fit
      return t_pred, loginvvar_pred

