"""
Homework4.
Replace 'pass' by your implementation.
"""
import sys
import math
import numpy as np
from scipy import ndimage
from scipy import optimize
from helper import refineF, visualize_3d


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    pts1 = pts1.astype('float64')
    pts2 = pts2.astype('float64')
    pts1 /= M
    pts2 /= M
    A = [pts2[:,0]*pts1[:,0], pts2[:,0]*pts1[:,1], pts2[:,0], pts2[:,1]*pts1[:,0], pts2[:,1]*pts1[:,1], pts2[:,1], pts1[:,0], pts1[:,1], np.ones_like(pts1[:,0])]
    A = np.asarray(A).T
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    U, S, V = np.linalg.svd(F)
    S[-1] = 0.0
    F = U.dot(np.diag(S)).dot(V)
    F = refineF(F, pts1, pts2)
    t = np.array([[1.0/M,0.0,0.0],[0.0,1.0/M,0.0],[0.0,0.0,1.0]])
    F = t.T.dot(F).dot(t)
    # np.savez('q2_1.npz', F=F, M=M)
    return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    pts1 = pts1.astype('float64')
    pts2 = pts2.astype('float64')
    pts1 /= M
    pts2 /= M
    A = [pts2[:,0]*pts1[:,0], pts2[:,0]*pts1[:,1], pts2[:,0], pts2[:,1]*pts1[:,0], pts2[:,1]*pts1[:,1], pts2[:,1], pts1[:,0], pts1[:,1], np.ones_like(pts1[:,0])]
    A = np.asarray(A).T
    U, S, V = np.linalg.svd(A)
    F1 = V[-1].reshape(3,3)
    F2 = V[-2].reshape(3,3)
    func = lambda x: np.linalg.det((x * F1) + ((1 - x) * F2))
    x0 = func(0)
    x1 = (2 * (func(1) - func(-1)) / 3.0) - ((func(2) - func(-2)) / 12.0)
    x2 = (0.5 * func(1)) + (0.5 * func(-1)) - func(0)
    x3 = func(1) - x0 - x1 - x2
    alphas = np.roots([x3,x2,x1,x0])
    alphas = np.real(alphas[np.isreal(alphas)])
    final_F = []
    for a in alphas:
        F = (a * F1) + ((1 - a) * F2)
        F = refineF(F, pts1, pts2)
        t = np.array([[1.0/M,0.0,0.0],[0.0,1.0/M,0.0],[0.0,0.0,1.0]])
        F = t.T.dot(F).dot(t)
        final_F.append(F)
    # np.savez('q2_2.npz', F=final_F[-1], M=M, pts1=pts1, pts2=pts2)
    return final_F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.T.dot(F).dot(K1)


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    projected_pts = []
    for i in range(len(pts1)):
        x1 = pts1[i,0]
        y1 = pts1[i,1]
        x2 = pts2[i,0]
        y2 = pts2[i,1]
        A = np.array([x1 * C1[2].T - C1[0].T, y1 * C1[2].T - C1[1].T, x2 * C2[2].T - C2[0].T, y2 * C2[2].T - C2[1].T])
        U, S, V = np.linalg.svd(A)
        ppts = V[-1]
        ppts = ppts / ppts[-1]
        projected_pts.append(ppts)
    projected_pts = np.asarray(projected_pts)
    projected_pts1 = C1.dot(projected_pts.T)
    projected_pts2 = C2.dot(projected_pts.T)
    projected_pts1 = projected_pts1 / projected_pts1[-1,:]
    projected_pts2 = projected_pts2 / projected_pts2[-1,:]
    projected_pts1 = projected_pts1[:2,:].T
    projected_pts2 = projected_pts2[:2,:].T
    error = np.linalg.norm(projected_pts1 - pts1, axis=-1)**2 + np.linalg.norm(projected_pts2 - pts2, axis=-1)**2
    error = error.sum()
    return projected_pts[:,:3], error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    epiline = F.dot(np.array([x1,y1,1]))
    a, b, c = epiline
    max_distance = 30
    window_size = 50
    im1_window = im1[y1 - window_size:y1 + window_size,x1 - window_size:x1 + window_size]
    im2_shape = im2.shape
    min_diff = sys.maxsize
    for y2 in range(y1 - max_distance,y1 + max_distance):
        x2 = -((b*y2) + c) / (a * 1.0)
        x2 = int(np.round(x2))
        if y2 - window_size >= 0 and y2 + window_size < im2_shape[0] and x2 - window_size >= 0 and x2 + window_size < im2_shape[1]:
            im2_window = im2[y2 - window_size:y2 + window_size,x2 - window_size:x2 + window_size]
            diff = np.abs(im1_window - im2_window)
            diff = ndimage.filters.gaussian_filter(diff, 7)
            diff = diff.sum()
            if diff < min_diff:
                min_diff = diff
                best_x2 = x2
                best_y2 = y2
    return best_x2, best_y2


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    num_iter = 200
    threshold = 1e-3
    total = len(pts1)
    max_inliers = 0
    for i in range(num_iter):
        idx = np.random.permutation(np.arange(total))[:7]
        selected_pts1, selected_pts2 = pts1[idx], pts2[idx]
        F7 = sevenpoint(selected_pts1, selected_pts2, M)
        for k, F in enumerate(F7):
            pts1_homo = np.concatenate((pts1, np.ones((pts1.shape[0],1))), axis=-1)
            pts2_homo = np.concatenate((pts2, np.ones((pts2.shape[0],1))), axis=-1)
            error = []
            for p, q in zip(pts1_homo, pts2_homo):
                error.append(q.T.dot(F).dot(p))
            error = np.abs(np.asarray(error))
            inliers = error < threshold
            # print (error)
            # print (inliers.sum(), max_inliers)
            if inliers.sum() > max_inliers:
                max_inliers = inliers.sum()
                best_inliers = inliers
                best_k = k
    selected_pts1, selected_pts2 = pts1[best_inliers], pts2[best_inliers]
    F = eightpoint(selected_pts1, selected_pts2, M)
    return F, best_inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r,2)
    I = np.eye(3)
    if theta == 0:
        R = I
    else:
        u = r/theta
        u_skew = np.array([[0, -u[2], u[1]],[u[2], 0, -u[0]],[-u[1], u[0], 0]])
        R = I * math.cos(theta) + (1 - math.cos(theta)) * (u.dot(u.T)) + (math.sin(theta) * u_skew)
    return R


def get_S(r):
    if np.linalg.norm(r,2) == np.pi and ((r[0] == r[1] and r[0] == 0 and r[1] == 0 and r[2] < 0) or  (r[0] == 0 and r[1] < 0) or (r[0] < 0)):
        ans = -1 * r
    else:
        ans = r
    return ans

def arcTan2(y,x):
    if x > 0:
        return np.arctan(y/x)
    elif x < 0:
        return np.pi + np.arctan(y/x)
    elif x == 0 and y < 0:
        return -np.pi/2
    elif x == 0 and y > 0:
        return np.pi/2

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    A = (R - R.T) / 2.0
    rho = np.array([A[2][1], A[0][2], A[1][0]])
    rho = np.reshape(rho, (rho.shape[0], 1))
    s = np.linalg.norm(rho, 2)
    c = (R[0][0] + R[1][1] + R[2][2] - 1.0) / 2.0
    if s == 0.0 and c == 1.0:
        r = np.zeros((3,1))
    elif s == 0.0 and c == -1.0:
        I = np.eye(3)
        temp = R + I
        v = temp[:,0]
        if len(np.nonzero(r1)) > 0:
            v = temp[:,0]
        elif len(np.nonzero(r2)) > 0:
            v = temp[:,1]
        elif len(np.nonzero(r3)) > 0:
            v = temp[:,2]
        u = v / np.linalg.norm(v,2)
        r = get_S(u * np.pi)
    elif s != 0.0:
        u = rho / s
        theta = arcTan2(s,c)
        r = u * theta
    return r

def reprojection_error(K1, M1, p1, K2, p2, x):
    N = len(p1)
    pts = x[:3*N]
    r = x[3*N:3*N+3]
    t = x[3*N+3:]
    t = np.reshape(t, (3,1))
    
    P = np.reshape(pts, (N,3))
    a = np.ones((1,N))
    P = np.vstack((P.T, a))
    
    R = rodrigues(r)
    M2 = np.hstack((R, t))

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    pts1_proj = np.matmul(C1, P)
    pts1_proj = pts1_proj / pts1_proj[2,:]

    pts2_proj = np.matmul(C2, P)
    pts2_proj = pts2_proj / pts2_proj[2,:]

    p1_hat = pts1_proj[:2,:].T
    p2_hat = pts2_proj[:2,:].T
    return p1_hat, p2_hat

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    p1_hat, p2_hat = reprojection_error(K1, M1, p1, K2, p2, x)
    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    return residuals


def rodriguesResidualCaller(x, K1, M1, p1, K2, p2):
    return rodriguesResidual(K1, M1, p1, K2, p2, x)

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    N = len(p1)
    r2 = invRodrigues(M2_init[:,:3])
    t2 = M2_init[:,-1]
    x = np.concatenate((P_init.flatten(), r2.flatten(), t2.flatten()))

    p1_hat, p2_hat = reprojection_error(K1, M1, p1, K2, p2, x)
    initial_error = np.linalg.norm(p1_hat - p1, axis=-1)**2 + np.linalg.norm(p2_hat - p2, axis=-1)**2
    initial_error = initial_error.sum()
    visualize_3d(P_init)

    final_x = optimize.least_squares(rodriguesResidualCaller, x, args=(K1, M1, p1, K2, p2))
    final_x = final_x.x

    p1_hat, p2_hat = reprojection_error(K1, M1, p1, K2, p2, final_x)
    final_error = np.linalg.norm(p1_hat - p1, axis=-1)**2 + np.linalg.norm(p2_hat - p2, axis=-1)**2
    final_error = final_error.sum()
    
    final_P = final_x[:3*N].reshape((N, 3))
    final_r = final_x[3*N:3*N+3]
    final_t = final_x[3*N+3:]
    final_R = rodrigues(final_r)
    final_M2 = np.hstack((final_R, final_t.reshape(3,1)))
    print ('Before optimization error: ', initial_error),
    print ('After optimization error: ', final_error)
    visualize_3d(final_P)

    return final_M2, final_P




