#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##dis光流法
import cv2 as cv
import numpy as np
cap = cv.VideoCapture(r'D:\data\Reid\cam_dong_420.mp4')
# https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
dis = cv.DISOpticalFlow_create()
while(1):
   ret, frame2 = cap.read()
   next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
   flow = dis.calc(prvs,next, None,)
   # flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
   mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
   hsv[...,0] = ang*180/np.pi/2   #angle弧度转角度
   hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX) #也可以用通道1来表示
   bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
   cv.imshow('result',bgr)
   cv.imshow('input', frame2)
   k = cv.waitKey(30) & 0xff
   if k == 27:
       break
   elif k == ord('s'):
       cv.imwrite('opticalfb.png',frame2)
       cv.imwrite('opticalhsv.png',bgr)
   prvs = next
cap.release()
cv.destroyAllWindows()


######LK光流法

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# cap = cv2.VideoCapture(r'D:\data\Reid\cam_dong_420.mp4')
# # params for ShiTomasi corner detection 特征点检测
# feature_params = dict( maxCorners = 10,
#                        qualityLevel = 0.1,
#                        minDistance = 10,
#                        blockSize = 3 )
#
# # Parameters for lucas kanade optical flow光流法参数
# lk_params = dict(winSize = (15,15),
#                   maxLevel = 0,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
# # Create some random colors 画轨迹
# color = np.random.randint(0,255,(100,3))
#
# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# roi = np.zeros_like(old_gray)
# x,y,w,h = 266,143,150,150
# roi[y:y+h, x:x+w] = 255
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = roi, **feature_params)
#
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
#
# while(1):
#     ret,frame = cap.read()
#     if not ret:
#       break
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # calculate optical flow
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#
#     # Select good points
#     good_new = p1[st==1]
#     good_old = p0[st==1]
#
#     # draw the tracks
#     for i,(new,old) in enumerate(zip(good_new,good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#         frame = cv2.circle(frame,(a,b),3,color[i].tolist(),-1)
#     img = cv2.add(frame,mask)
#
#     cv2.imshow('frame',img)
#     key = cv2.waitKey(60) & 0xff
#     if key == 27:  # 按下ESC时，退出
#         break
#     elif key == ord(' '):  # 按下空格键时，暂停
#         cv2.waitKey(0)
#
#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)
#
# cv2.destroyAllWindows()
# cap.release()
