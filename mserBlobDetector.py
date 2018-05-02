import numpy as np
import cv2 as cv


img = cv.imread("C:/Users/mobileRigg/Desktop/guardian/imgen-master/targets/t_6_8.png")
mser = cv.MSER_create(8,60,2000,0.09,0.5,200,1.01,0.003,5)
# https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
vis = img.copy()

regions, _ = mser.detectRegions(gray)
hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv.polylines(vis, hulls, 1, (0, 255, 0))


cv.imshow('img', vis)

