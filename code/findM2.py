'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt

import submission as sub
from helper import camera2


def main():
	data = np.load('../data/some_corresp.npz')
	im1 = plt.imread('../data/im1.png')
	im2 = plt.imread('../data/im2.png')

	N = data['pts1'].shape[0]
	M = 640

	intrinsics = np.load('../data/intrinsics.npz')
	K1 = intrinsics['K1']
	K2 = intrinsics['K2']

	F = sub.eightpoint(data['pts1'], data['pts2'], M)
	E = sub.essentialMatrix(F, K1, K2)
	M1 = np.zeros((3,4))
	M1[0,0] = 1
	M1[1,1] = 1
	M1[2,2] = 1
	M2s = camera2(E)
	C1 = K1.dot(M1)
	
	for i in range(4):
		M2 = M2s[:,:,i]
		C2 = K2.dot(M2)
		P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
		if (P[:,-1] >= 0.0).all():
			break

	np.savez('q3_3.npz', M2=M2, C2=C2, P=P)

if __name__ == '__main__':
	main()
