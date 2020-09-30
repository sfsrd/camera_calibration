#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob


def points3Dto2D(x, y, z, rvecs, tvecs, mtx, dist):
    axis = np.float32([x, y, z]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, np.asarray(rvecs), np.asarray(tvecs), mtx, dist)
    imgpts = np.asarray(imgpts)
    imgpts = np.reshape(imgpts, 2)
    return imgpts


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





