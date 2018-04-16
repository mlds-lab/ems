
import numpy as np
from numpy import linalg as LA


def find_pattern(Q, n_patterns):
    """
    Function to find the top missing value patterns in data by counting.

    Args:
        Q (numpy array): missing value indicator array of shape (d1, d2)
        n_patterns (int): number of top missing value patterns

    Returns:
        P (numpy array): array of shape (n_patterns, d2) including top missing value patterns

    """

    # Q = convert_binary(Q)
    P, counts = np.unique(Q, axis=0, return_counts=True)
    indices = np.argsort(counts)[-n_patterns:]
    P = P[indices, :]

    return P


def fit_pattern(Q, p):
    """
    Function to find all the data cases with patterns in Q that have pattern p as subset.

    Args:
        Q (numpy array): missing value indicator array of shape (d1, d2).
        p (numpy array): pattern to match of shape (d2, ).

    Returns:
        indices (numpy array): indices of data cases in Q matching pattern p.

    """

    d1, d2 = np.shape(Q)
    # Q = convert_binary(Q)
    indices = np.where(np.sum(np.multiply(np.tile(p, (d1, 1)), Q), axis=1) == np.sum(p))[0]

    # find all the data cases with patterns in Q that exactly match the pattern p:
    # indices = np.where(LA.norm(Q - np.tile(p, (d1, 1)), 1, axis=1) == 0)[0]

    return indices


def predict_pattern(P, q):
    """
    Function to find the closest pattern from P to data case pattern q.

    Args:
        P (numpy array): array of top missing value patterns to match of shape (d1, d2).
        q (numpy array): data case pattern; array of shape (d2, ).

    Returns:
        pattern_index (int): index of pattern in P that best matched given data case pattern q.

    """

    d1, d2 = np.shape(P)
    pattern_index = np.argmin(LA.norm(P - np.tile(q, (d1, 1)), 1, axis=1))

    # find the closest pattern from P that includes the data case pattern q.
    # pattern_index = np.argmin(LA.norm(P - np.tile(q, (d1, 1)) > 0, 1, axis=1))

    return pattern_index


def convert_binary(Q):

    Q1 = np.zeros(Q.shape)
    Q1[Q > 0] = 1

    return Q1
