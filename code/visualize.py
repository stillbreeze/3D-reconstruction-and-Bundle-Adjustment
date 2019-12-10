'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt

import submission as sub
from helper import camera2, visualize_3d

def main():
	data = np.load('../data/some_corresp.npz')
	im1 = plt.imread('../data/im1.png')
	im2 = plt.imread('../data/im2.png')

	N = data['pts1'].shape[0]
	M = 640

	intrinsics = np.load('../data/intrinsics.npz')
	temple_pts = np.load('../data/templeCoords.npz')
	x1, y1 = temple_pts['x1'], temple_pts['y1']
	x1 = np.squeeze(x1)
	y1 = np.squeeze(y1)
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

	x2 = []
	y2 = []
	for i, j in zip(x1, y1):
		k, l = sub.epipolarCorrespondence(im1, im2, F, i, j)
		x2.append(k)
		y2.append(l)
	x2 = np.asarray(x2)
	y2 = np.asarray(y2)
	
	for i in range(4):
		M2 = M2s[:,:,i]
		C2 = K2.dot(M2)
		P, err = sub.triangulate(C1, np.array([x1, y1]).T, C2, np.array([x2, y2]).T)
		if (P[:,-1] >= 0.0).all():
			break
	visualize_3d(P)
	np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

if __name__ == '__main__':
	main()