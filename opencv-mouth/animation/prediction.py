import numpy as np
import numpy.linalg as la
# import torch
import cv2
import imutils
import pandas as pd
from detect_face_parts import detect_face_parts
# from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
import scipy.stats as stats

def minmax(a):
  big = a.max()
  small = a.min()
  std = (a-small)/(big-small)
  h = 1
  l = -1
  return std*(h-l)+l

def find_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    Px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    Py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    return np.array([Px / D, Py / D])
  
def transform_to_standard(im):
  mid = find_intersection(im[0],im[4],im[2],im[6])
  im = (im-mid)
  yx_ratio = la.norm(im[2]) / la.norm(im[4])
  B = la.inv(np.vstack((im[4],im[2])))
  im = im @ B
  im[:,1]*=yx_ratio
  im[0] = (-1,0)
  im[:,0] -= im[2,0]
  return im

def sq_error(im1,im2):
  return ((im1[:,1]-im2[:,1])**2).sum()

keypoints = pd.read_csv('./keypoints.csv').to_numpy()
agape_points = transform_to_standard(keypoints[:,0:2])
close_points = transform_to_standard(keypoints[:,2:4])
open_points = transform_to_standard(keypoints[:,4:6])
# An insane amount of bullshit lmfao for flipping the open image upside down but keeping corresponding points
tmp = open_points[(1,2,3),:].copy()
open_points[(1,2,3),:]=open_points[(7,6,5),:] # Flip the open
open_points[(7,6,5),:]=tmp
# Scaling hyperparameters (can't be negative without special stuff^^^)
close_points[3,1] = close_points[1,1]
close_points[2,1] = close_points[1,1]
close_points[:,1]*=1.5
agape_points[:,1]*=0.7
open_points[:,1]*=-0.3

def predict(frame):
  loss = sq_error
  frame = imutils.resize(frame, width=500)
  output = detect_face_parts(frame)
  # cv2_imshow(output)
  if output is None:
    return None
  # cv2.imshow('image', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    pass
  try:
    output = transform_to_standard(output)
  except:
    print("No inverse")
    return None
  output[:,1]*=0.7
  #TODO: vectorize
  scores = [loss(output,close_points),loss(output,agape_points),loss(output,open_points)]
  pred = np.argmin(scores)
  return pred

