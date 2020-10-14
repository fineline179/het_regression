# functions/classes from https://github.com/amber-kshz/PRML repo notebook 3

import numpy as np
from scipy.linalg import eigh
from matplotlib import pyplot as plt


class BayesianRidgeRegression:

  def __init__(self, alpha=1.0, beta=1.0):
    self.alpha = alpha
    self.beta = beta
    self.m = None  # posterior mean
    self.S = None  # posterior covariancematrix

  def calc_posterior_params(self, Phi, t):
    """
    This method calculates posterior mean and covariance matrix from the training
    data Phi and t.

    Parameters
    ----------
    Phi : 2-D numpy array
        (N,M) array, representing design matrix
    t : 1-D numpy array
        (N,) array, representing target values
    """
    self.S = np.linalg.inv(
      self.alpha * np.identity(len(Phi[0])) + self.beta * (Phi.T) @ Phi)
    self.m = self.beta * (self.S @ (Phi.T) @ t)

  def predict(self, Phi, return_std=False):
    """
    This method makes prediction on the input Phi, and returns predictive mean (and
    standard deviation)

    Parameters
    ----------
    Phi : 2-D numpy array
        (N_test, M) numpy array. M must be equal to "M" (the length in the second
        dimension) of the training data.
    return_std : boolean, default False
        If True, the method also returns predictive standard deviation

    Returns
    ----------
    pred_mean : 1-D numpy array
        (N_test,) numpy array representing predictive mean
    pred_std : 1-D numpy array
        (N_test,) numpy array representing predictive mean
    """
    pred_mean = Phi @ self.m
    if not (return_std):
      return pred_mean
    else:
      pred_std = np.sqrt(1.0 / self.beta + np.diag(Phi @ self.S @ (Phi.T)))
      return pred_mean, pred_std

  def calc_evidence(self, Phi, t):
    """
    This method calculates the evidence with respect to the data Phi and t

    Parameters
    ----------
    Phi : 2-D numpy array
        (N,M) array, representing design matrix
    t : 1-D numpy arra
        (N,) array, representing target values

    Returns
    ----------
    evidence : float

    """
    N, M = np.shape(Phi)
    evidence = 0.5 * M * np.log(self.alpha) + 0.5 * N * np.log(self.beta) \
               - 0.5 * self.beta * np.linalg.norm(
      t - Phi @ self.m)**2 - 0.5 * self.alpha * (self.m @ self.m) \
               - 0.5 * np.log(
      np.linalg.det(self.alpha * np.identity(M) + self.beta * (Phi.T) @ Phi)) \
               - 0.5 * N * np.log(2 * np.pi)
    return evidence

  def empirical_bayes(self, Phi, t, tol, maxiter, show_message=True):
    """
    This method performs empirical bayes (or evidence approximation),
    where hyper parameters alpha and beta are chosen in such a way that they maximize
    the evidence.

    Parameters
    ----------
    Phi : 2-D numpy array
        (N,M) array, representing design matrix
    t : 1-D numpy arra
        (N,) array, representing target values
    tol : float
        The tolerance.
        If the changes of alpha and beta are smaller than the value, the iteration is
        judged as converged.
    maxiter : int
        The maximum number of iteration
    show_message : boolean, default True
        If True, the message indicating whether the optimization terminated
        successfully is shown.
    """
    tmp_lambdas = eigh((Phi.T) @ Phi)[0]
    cnt = 0
    while cnt < maxiter:
      lambdas = self.beta * tmp_lambdas
      self.calc_posterior_params(Phi, t)

      alpha_old = self.alpha
      beta_old = self.beta

      gamma = np.sum(lambdas / (self.alpha + lambdas))
      self.alpha = gamma / np.dot(self.m, self.m)
      self.beta = (len(t) - gamma) / (np.linalg.norm(t - Phi @ self.m)**2)
      if (abs(self.alpha - alpha_old) < tol) and (abs(self.beta - beta_old) < tol):
        break
      cnt += 1
    if show_message:
      if cnt <= maxiter:
        print(f"Optimization terminated succesfully. The number of iteration : {cnt}")
      else:
        print("Maximum number of iteration exceeded.")

  def fit(self, Phi, t, tol=1e-4, maxiter=100, show_message=True,
          optimize_hyperparams=False):
    """
    This method performs fitting.
    The user can choose whether or not to perform empirical Bayes.

    Parameters
    ----------
    Phi : 2-D numpy array
        (N,M) array, representing design matrix
    t : 1-D numpy arra
        (N,) array, representing target values
    tol : float
        The tolerance.
        If the changes of alpha and beta are smaller than the value, the iteration is
        judged as converged.
    maxiter : int
        The maximum number of iteration
    show_message : boolean, default True
        If True, the message indicating whether the optimization terminated
        successfully is shown.
    optimize_hyperparams : boolean, default False
        If True, the hyper parameters alpha and beta are optimized by empirical Bayes.
    """
    if optimize_hyperparams:
      self.empirical_bayes(Phi, t, tol, maxiter, show_message)
    self.calc_posterior_params(Phi, t)
