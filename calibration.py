#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import math

def points3Dto2D(x, y, z, rvecs, tvecs, mtx, dist):
    axis = np.float32([x, y, z]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, np.asarray(rvecs), np.asarray(tvecs), mtx, dist)
    imgpts = np.asarray(imgpts)
    imgpts = np.reshape(imgpts, 2)
    return imgpts

def calculateError(objpoints, rvecs, tvecs, mtx, dist):
	mean_error = 0
	for i in range(len(objpoints)):
		imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		mean_error += error
	return mean_error/len(objpoints)

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
	#x - roll, y - pitch, z - yaw
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        roll = math.atan2(R[2,1] , R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else :
        roll = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

# Определение размеров шахматной доски
CHECKERBOARD = (5,8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Создание вектора для хранения векторов трехмерных точек для каждого изображения шахматной доски
objpoints = []
# Создание вектора для хранения векторов 2D точек для каждого изображения шахматной доски
imgpoints = [] 


# Определение мировых координат для 3D точек
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


images = glob.glob('./images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (300,300))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
h,w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)


a = points3Dto2D(3, 0, 0, rvecs, tvecs, mtx, dist)
print(a)

#saving to npy format
with open('intrinsic_matrix.npy', 'wb') as f:
    np.save(f, np.array(mtx))

with open('dist.npy', 'wb') as f:
    np.save(f, np.array(dist))

with open('rvecs.npy', 'wb') as f:
    np.save(f, np.array(rvecs))

with open('tvecs.npy', 'wb') as f:
    np.save(f, np.array(tvecs))

total_error = calculateError(objpoints, rvecs, tvecs, mtx, dist)
print('total_error: ', total_error)

rmat = cv2.Rodrigues(np.asarray(rvecs))[0]
print('Rmat: ', rmat)
print('Angles: ', rotationMatrixToEulerAngles(rmat))