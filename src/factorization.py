from logging import getLogger
import os
import numpy as np
import cv2
import copy


logger = getLogger(__name__)


def factorization(coords_array):
    frame_num = coords_array.shape[0]

    w = coords_array.transpose(2, 0, 1).reshape(-1, coords_array.shape[1])
    w_bar = w - w.mean(axis=1)[:, None]

    U, S, V = np.linalg.svd(w_bar, full_matrices=False)
    S = np.diag(S[:3])
    U = U[:, :3]
    V = V[:3, :]

    right = S**0.5 @ V
    left = U @ S**0.5

    left_x = left[:frame_num]
    left_y = left[frame_num:]

    A = np.empty((2*frame_num, 6))
    for i in range(frame_num):
        A[2*i, 0] = (left_x[i, 0]**2) - (left_y[i, 0]**2)
        A[2*i, 1] = ((left_x[i, 0] * left_x[i, 1]) - (left_y[i, 0] * left_y[i, 1])) * 2
        A[2*i, 2] = ((left_x[i, 0] * left_x[i, 2]) - (left_y[i, 0] * left_y[i, 2])) * 2
        A[2*i, 3] = (left_x[i, 1]**2) - (left_y[i, 1]**2)
        A[2*i, 5] = (left_x[i, 2]**2) - (left_y[i, 2]**2)
        A[2*i, 4] = ((left_x[i, 2] * left_x[i, 1]) - (left_y[i, 2] * left_y[i, 1])) * 2

        A[2*i + 1, 0] = left_x[i, 0] * left_y[i, 0]
        A[2*i + 1, 1] = left_x[i, 1] * left_y[i, 0] + left_x[i, 0] * left_y[i, 1]
        A[2*i + 1, 2] = left_x[i, 2] * left_y[i, 0] + left_x[i, 0] * left_y[i, 2]
        A[2*i + 1, 3] = left_x[i, 1] * left_y[i, 1]
        A[2*i + 1, 4] = left_x[i, 2] * left_y[i, 1] + left_x[i, 1] * left_y[i, 2]
        A[2*i + 1, 5] = left_x[i, 2] * left_y[i, 2]
    _, _, V = np.linalg.svd(A, full_matrices=False)
    v = V[-1].T

    A = np.array([
        [v[0], v[1], v[2]],
        [v[1], v[3], v[4]],
        [v[2], v[4], v[5]]
    ])
    U, S, _ = np.linalg.svd(A)
    left = U @ np.diag(S**0.5)

    return left @ right
