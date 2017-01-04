from __future__ import print_function

import numpy as np

def eigenvalue_features(segments):
  # input should be a list of segments ( 2D numpy array of points )
  assert type(segments) == list
  assert type(segments[0]) == np.ndarray
  # Output
  features_vectors = []

  n_segments = len(segments)
  for segment in segments:
    ## Compute covariance matrix and eigenvalues
    n_points = segment.shape[0]
    # Find mean of points in each dimension
    means = segment.mean(axis=0)
    variances = segment - means
    # Compute covariance matrix (symmetric)
    covariance_matrix = np.zeros((3,3))
    for k in range(0,6):
      i = [0,0,0,1,1,2][k]
      j = [0,1,2,1,2,2][k]
      covariance = np.mean(variances[:,i]*variances[:,j])
      covariance_matrix[i,j] = covariance
      covariance_matrix[j,i] = covariance
    # Compute eigenvalues of covariance matrix
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    # Sort eigenvalues from smallest to largest
    eigenvalues.sort()
    normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)
    e1 = normalized_eigenvalues[2]
    e2 = normalized_eigenvalues[1]
    e3 = normalized_eigenvalues[0]

    ## Compute features
    n_features = 7
    features = np.zeros(n_features)
    # Linearity
    features[0] = ( e1 - e2 ) / e1
    # Planarity
    features[1] = ( e2 - e3 ) / e1
    # Scattering
    features[2] = e3 / e1
    # Omnivariance
    features[3] = np.power(e1 * e2 * e3, 1.0/3.0)
    # Anisotropy
    features[4] = (e1 - e3) / e1
    # Eigenentropy
    features[5] = - e1 * np.log(e1) - e2 * np.log(e2) - e3 * np.log(e3)
    # Change of Curvature
    features[6] = e3

    features_vectors.append( features )

  return np.array(features_vectors)
