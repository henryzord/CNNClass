# encoding=utf-8

import numpy as np
from scipy.spatial.distance import cdist

__author__ = 'Henry Cagnini'


def hausdorff(C1, C2):
    assert type(C2) is np.ndarray, "A is not a numpy.ndarray!"
    assert type(C1) is np.ndarray, "B is not a numpy.ndarray!"

    D = cdist(C1, C2, 'euclidean')

    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    return max(H1, H2)


def main():
    # first case
    A = np.array([1, 2, 3])[:, np.newaxis]
    B = np.array([4, 5, 6])[:, np.newaxis]
    C = np.array([4, 5, 20])[:, np.newaxis]

    print hausdorff(A, B)
    print hausdorff(A, C)

if __name__ == '__main__':
    main()
