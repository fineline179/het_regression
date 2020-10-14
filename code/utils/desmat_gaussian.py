import numpy as np

def gen_desmat_gaussian(X, params):
  """
  Generates design matrix with Gaussian basis functions.

  Args:
    X ((N,) array): input data
    params (dict):
      params['mus'] : (M-1,)  array. params['mus'][j] being the center of the
                      jth Gaussian
      params['s'] : double. positive real number, which stands for the width of
                    Gaussians
  Returns:
    Phi((N, M) array): design matrix, with Phi[n, m] = $\phi_m(x_n)$
  """

  s = params['s']
  mus = params['mus']
  # reshape X and mus into column vectors
  X = np.reshape(X, (len(X), 1) )
  mus = np.reshape(mus, (len(mus), 1) )
  Phi = np.zeros((len(X), len(mus) + 1) )
  Phi[:, 0] = np.ones(len(X))  # the 0th basis fn is a constant
  A = (-2 * (X @ mus.T) + X**2) + np.reshape(mus**2, len(mus))
  Phi[:, 1:] = np.exp(-A / (2 * s * s))
  return Phi
