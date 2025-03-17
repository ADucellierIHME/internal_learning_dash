import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg
from statistics import covariance

def get_margin_matrix_vector(v_i, v_j, mu_i, mu_j):
    """
    In 2D, transform the I + J conditions on the margins
    into a matrix formulation
    """
    assert isinstance(v_i, np.ndarray), \
        'Coefficients for the margin in the first dimension should be a Numpy array.'
    assert isinstance(v_j, np.ndarray), \
        'Coefficients for the margin in the second dimension should be a Numpy array.'
    assert isinstance(mu_i, np.ndarray), \
        'Values of the margin in the first dimension should be a Numpy array.'
    assert isinstance(mu_j, np.ndarray), \
        'Values of the margin in the second dimension should be a Numpy array.'
    assert len(v_i) == len(mu_j), \
        'Coefficients in the first dimension and margin values in the second dimension should have the same size.'
    assert len(v_j) == len(mu_i), \
        'Coefficients in the second dimension and margin values in the first dimension should have the same size.'
    if np.sum(mu_i) != 0.0:
        assert abs((np.sum(mu_i) - np.sum(mu_j)) /  np.sum(mu_i)) < 1e-2, \
            'Sum of margins over rows and columns should be equal.'
    elif np.sum(mu_j) != 0.0:
        assert abs((np.sum(mu_i) - np.sum(mu_j)) /  np.sum(mu_j)) < 1e-2, \
            'Sum of margins over rows and columns should be equal.'

    I = len(mu_i)
    J = len(mu_j)
    A = np.zeros((I + J, I * J))
    y = np.zeros(I + J)
    # Partial sums in the first dimension
    for i in range(0, I):
        for j in range(0, J):
            A[i, j * I + i] = A[i, j * I + i] + v_i[j]
            A[I + j, j * I + i] = A[I + j, j * I + i] + v_j[i]
    for i in range(0, I):
        y[i] = mu_i[i]
    for j in range(0, J):
        y[I + j] = mu_j[j]
    return (A, y)

def raking_chi2_distance(x, q, A, y, direct=True):
    """
    Raking using the chi2 distance (mu - x)^2 / 2x.
    Input:
      x: 1D Numpy array, observed values x_n = x_i,j if n = j I + i
      q: 1D Numpy array, weights for the observations
      A: 2D Numpy array, linear constraints
      y: 1D Numpy array, partial sums
      direct: boolean, if True, we solve for lambda and compute mu;
          if False, we solve directly for lambda and mu
    Output:
      mu: 1D Numpy array, raked values
      lambda_k: 1D Numpy array, dual
    """
    assert isinstance(x, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q, np.ndarray), \
        'Weights should be a Numpy array.'
    assert len(x) == len(q), \
        'Observations and weights arrays should have the same size.'
    assert isinstance(A, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert isinstance(y, np.ndarray), \
        'Partial sums should be a Numpy array.'
    assert np.shape(A)[0] == len(y), \
        'The number of linear constraints should be equal to the number of partial sums.'
    assert np.shape(A)[1] == len(x), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    M = np.shape(A)[0]
    N = np.shape(A)[1]
    if direct:
        y_hat = np.matmul(A, x)
        Phi = np.matmul(A, np.transpose(A * x * q))
        # Compute Moore-Penrose pseudo inverse to solve the system
        U, S, Vh = np.linalg.svd(Phi, full_matrices=True)
        V = np.transpose(Vh)
        # Invert diagonal matrix while dealing with 0 and near 0 values
        Sdiag = np.diag(S)
        Sdiag[np.abs(Sdiag) <= 1.0e-12] = 1.0e-12
        Sinv = 1.0 / Sdiag
        Sinv[np.abs(Sdiag) <= 1.0e-12] = 0.0
        Phi_plus = np.matmul(V, np.matmul(Sinv, np.transpose(U)))
        lambda_k = np.matmul(Phi_plus, y_hat - y)
        mu = x * (1 - q * np.matmul(np.transpose(A), lambda_k))
    else:
        Phi = np.concatenate((np.concatenate((np.diag(1.0 / (q * x)), \
            np.transpose(A)), axis=1), \
            np.concatenate((A, np.zeros((M, M))), axis=1)), axis=0)
        b = np.concatenate((1.0 / q, y), axis=0)
        # Compute Moore-Penrose pseudo inverse to solve the system
        U, S, Vh = np.linalg.svd(Phi, full_matrices=True)
        V = np.transpose(Vh)
        # Invert diagonal matrix while dealing with 0 and near 0 values
        Sdiag = np.diag(S)
        Sdiag[np.abs(Sdiag) <= 1.0e-12] = 1.0e-12
        Sinv = 1.0 / Sdiag
        Sinv[np.abs(Sdiag) <= 1.0e-12] = 0.0
        Phi_plus = np.matmul(V, np.matmul(Sinv, np.transpose(U)))
        result = np.matmul(Phi_plus, b)
        mu = result[0:N]
        lambda_k = result[N:(N + M)]
    return (mu, lambda_k)

def compute_variance(mu_0, lambda_0, x, q, A, method, alpha=0, l=0, h=1):
    """
    """
    M = np.shape(A)[0]
    N = np.shape(A)[1]

    # Derivatives of distance function
    if method == 'general':
        H1_mu_diag = np.zeros(len(mu_0))
        H1_mu_diag[x!=0] = np.power(mu_0[x!=0] / x[x!=0], alpha - 1) / (q[x!=0] * x[x!=0])
        H1_mu_diag[x==0] = 0.0
        H1_mu = np.diag(H1_mu_diag)
        H1_x_diag = np.zeros(len(x))
        H1_x_diag[mu_0!=0] = - np.power(mu_0[mu_0!=0] / x[mu_0!=0], alpha + 1) / (q[mu_0!=0] * mu_0[mu_0!=0])
        H1_x_diag[mu_0==0] = 0.0
        H1_x = np.diag(H1_x_diag)
    elif method == 'l2':
        H1_mu = np.identity(len(mu_0))
        H1_x = - np.identity(len(x))
    elif method == 'logit':
        H1_mu_diag = np.zeros(len(mu_0))
        H1_mu_diag[(mu_0!=l)&(mu_0!=h)] = 1.0 / (mu_0[(mu_0!=l)&(mu_0!=h)] - l[(mu_0!=l)&(mu_0!=h)]) + \
                                          1.0 / (h[(mu_0!=l)&(mu_0!=h)] - mu_0[(mu_0!=l)&(mu_0!=h)])
        H1_mu_diag[(mu_0==l)|(mu_0==h)] = 0.0
        H1_mu = np.diag(H1_mu_diag)
        H1_x_diag = np.zeros(len(x))
        H1_x_diag[(x!=l)&(x!=h)] = - 1.0 / (x[(x!=l)&(x!=h)] - l[(x!=l)&(x!=h)]) - \
                                     1.0 / (h[(x!=l)&(x!=h)] - x[(x!=l)&(x!=h)])
        H1_x_diag[(x==l)|(x==h)] = 0.0
        H1_x = np.diag(H1_x_diag)

    # Gradient with respect to mu and lambda
    H1_lambda = np.transpose(np.copy(A))
    H2_mu = np.copy(A)
    H2_lambda = np.zeros((M, M))    
    DH_mu_lambda = np.concatenate(( \
        np.concatenate((H1_mu, H1_lambda), axis=1), \
        np.concatenate((H2_mu, H2_lambda), axis=1)), axis=0)

    # Gradient with respect to x and y
    H1_y = np.zeros((N, M))
    H2_x = np.zeros((M, N))
    H2_y = - np.identity(M)    
    DH_x_y = np.concatenate(( \
        np.concatenate((H1_x, H1_y), axis=1), \
        np.concatenate((H2_x, H2_y), axis=1)), axis=0)

    # Compute Moore-Penrose pseudo inverse of D_mu_lambda
    U, S, Vh = np.linalg.svd(DH_mu_lambda, full_matrices=True)
    V = np.transpose(Vh)
    Sdiag = np.diag(S)
    Sdiag[np.abs(Sdiag) <= 1.0e-12] = 1.0e-12
    Sinv = 1.0 / Sdiag
    Sinv[np.abs(Sdiag) <= 1.0e-12] = 0.0
    DH_mu_lambda_plus = np.matmul(V, np.matmul(Sinv, np.transpose(U)))
        
    # Gradient of mu and lambda with respect to x and y
    Dh = - np.matmul(DH_mu_lambda_plus, DH_x_y)
    dh_x = Dh[0:N, 0:N]
    dh_y = Dh[0:N, N:(N + M)]
    return (dh_x, dh_y)

def rake_mean(I, J):
    # Define categorical variables
    X1 = np.tile(np.arange(1, I + 1), J)
    X2 = np.repeat(np.arange(1, J + 1), I)
    margins = ['X1_' + str(i) for i in range(1, I + 1)] + \
              ['X2_' + str(j) for j in range(1, J + 1)]

    # Generate balanced table
    rng = np.random.default_rng(0)
    mu_ij = rng.uniform(low=2.0, high=3.0, size=(I, J))
    mu_i0 = np.sum(mu_ij, axis=1)
    mu_0j = np.sum(mu_ij, axis=0)

    # Add noise to the data and generate samples from the MVN
    mean = mu_ij.flatten(order='F') + rng.normal(0.0, 0.1, size=I*J)
    cov = 0.01 * np.ones((I * J, I * J))
    np.fill_diagonal(cov, np.arange(0.01, 0.01 * (I * J + 1), 0.01))

    # Define constraints for the raking
    (A, s) = get_margin_matrix_vector(np.repeat(1, J), np.repeat(1, I), mu_i0, mu_0j)

    # Rake the mean
    (beta0, lambda_k) = raking_chi2_distance(mean, np.ones(I * J), A, s)
    df_raked = pd.DataFrame({'X1': X1, \
                             'X2': X2, \
                             'observations': mean, \
                             'raked_values': beta0})

    # Compute the gradient
    (dh_x, dh_y) = compute_variance(beta0, lambda_k, mean, np.ones(I * J), A, 'general', 1)
    df_x = []
    df_y = []
    for i in range(0, I * J):
        df_x.append(pd.DataFrame({'raked_1': np.repeat(X1[i], I * J), \
                                  'raked_2': np.repeat(X2[i], I * J), \
                                  'X1': X1, \
                                  'X2': X2, \
                                  'grad_x': dh_x[i, :]}))
        df_y.append(pd.DataFrame({'raked_1': np.repeat(X1[i], I + J), \
                                  'raked_2': np.repeat(X2[i], I + J), \
                                  'margins': margins, \
                                  'grad_y': dh_y[i, :]}))
    df_x = pd.concat(df_x)
    df_y = pd.concat(df_y)

    # Compute the covariance
    dh_x = dh_x.flatten()
    dh_y = dh_y.flatten()
    covariance_mean = np.zeros((I * J, I * J))
    for i in range(0, I * J):
        for j in range(0, I * J):
            dh_x_i = dh_x[(I * J * i):(I * J * i + I * J)]
            dh_y_i = dh_y[((I + J) * i):((I + J) * i + (I + J))]
            dh_i = np.concatenate((dh_x_i, dh_y_i), axis=0)
            dh_x_j = dh_x[(I * J * j):(I * J * j + I * J)]
            dh_y_j = dh_y[((I + J) * j):((I + J) * j + (I + J))]
            dh_j = np.concatenate((dh_x_j, dh_y_j), axis=0)
            sigma = np.concatenate(( \
                np.concatenate((cov, np.zeros((I * J, (I + J)))), axis=1), \
                np.concatenate((np.zeros(((I + J), I * J)), np.zeros(((I + J), (I + J)))), axis=1)), axis=0)
            covariance_mean[i, j] = np.matmul(np.transpose(dh_i), np.matmul(sigma, dh_i))

    return (X1, X2, df_raked, df_x, df_y, covariance_mean)

def rake_draws(I, J, num_samples):
    # Define categorical variables
    X1 = np.tile(np.arange(1, I + 1), J)
    X2 = np.repeat(np.arange(1, J + 1), I)
    margins = ['X1_' + str(i) for i in range(1, I + 1)] + \
              ['X2_' + str(j) for j in range(1, J + 1)]

    # Generate balanced table
    rng = np.random.default_rng(0)
    mu_ij = rng.uniform(low=2.0, high=3.0, size=(I, J))
    mu_i0 = np.sum(mu_ij, axis=1)
    mu_0j = np.sum(mu_ij, axis=0)

    # Add noise to the data and generate samples from the MVN
    mean = mu_ij.flatten(order='F') + rng.normal(0.0, 0.1, size=I*J)
    cov = 0.01 * np.ones((I * J, I * J))
    np.fill_diagonal(cov, np.arange(0.01, 0.01 * (I * J + 1), 0.01))

    # Define constraints for the raking
    (A, s) = get_margin_matrix_vector(np.repeat(1, J), np.repeat(1, I), mu_i0, mu_0j)

    # Rake the mean
    (beta0, lambda_k) = raking_chi2_distance(mean, np.ones(I * J), A, s)
    df_raked = pd.DataFrame({'X1': X1, \
                             'X2': X2, \
                             'observations': mean, \
                             'raked_values': beta0})

    # Compute the gradient
    (dh_x, dh_y) = compute_variance(beta0, lambda_k, mean, np.ones(I * J), A, 'general', 1)
    df_x = []
    df_y = []
    for i in range(0, I * J):
        df_x.append(pd.DataFrame({'raked_1': np.repeat(X1[i], I * J), \
                                  'raked_2': np.repeat(X2[i], I * J), \
                                  'X1': X1, \
                                  'X2': X2, \
                                  'grad_x': dh_x[i, :]}))
        df_y.append(pd.DataFrame({'raked_1': np.repeat(X1[i], I + J), \
                                  'raked_2': np.repeat(X2[i], I + J), \
                                  'margins': margins, \
                                  'grad_y': dh_y[i, :]}))
    df_x = pd.concat(df_x)
    df_y = pd.concat(df_y)

    # Compute the covariance
    dh_x = dh_x.flatten()
    dh_y = dh_y.flatten()
    covariance_mean = np.zeros((I * J, I * J))
    for i in range(0, I * J):
        for j in range(0, I * J):
            dh_x_i = dh_x[(I * J * i):(I * J * i + I * J)]
            dh_y_i = dh_y[((I + J) * i):((I + J) * i + (I + J))]
            dh_i = np.concatenate((dh_x_i, dh_y_i), axis=0)
            dh_x_j = dh_x[(I * J * j):(I * J * j + I * J)]
            dh_y_j = dh_y[((I + J) * j):((I + J) * j + (I + J))]
            dh_j = np.concatenate((dh_x_j, dh_y_j), axis=0)
            sigma = np.concatenate(( \
                np.concatenate((cov, np.zeros((I * J, (I + J)))), axis=1), \
                np.concatenate((np.zeros(((I + J), I * J)), np.zeros(((I + J), (I + J)))), axis=1)), axis=0)
            covariance_mean[i, j] = np.matmul(np.transpose(dh_i), np.matmul(sigma, dh_i))

    rng = np.random.default_rng(0)
    x = rng.multivariate_normal(mean, cov, num_samples)
    mu = np.zeros((num_samples, I * J))
    for n in range(0, num_samples):
        x_n = x[n, :]
        (mu_n, lambda_k) = raking_chi2_distance(x_n, np.ones(I * J), A, s)
        mu[n, :] = mu_n
    mean_draws = np.mean(mu, 0)
    covariance_draws = np.zeros((I * J, I * J))
    for i in range(0, I * J):
        for j in range(0, I * J):
            covariance_draws[i, j] = covariance(mu[:, i], mu[:, j])

    return (X1, X2, df_raked, mean_draws, covariance_mean, covariance_draws)

