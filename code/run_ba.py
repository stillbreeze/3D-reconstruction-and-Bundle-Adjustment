import numpy as np
import matplotlib.pyplot as plt
import submission as sub

from helper import displayEpipolarF, chooseSevenPoints, epipolarMatchGUI, getBaParams

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640

data_noisy = np.load('../data/some_corresp_noisy.npz')
K1, K2, M1, M2, P, inliers = getBaParams(data_noisy, M)
M2, P = sub.bundleAdjustment(K1, M1, data_noisy['pts1'][inliers], K2, M2, data_noisy['pts2'][inliers], P)