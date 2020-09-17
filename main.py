""" entry point to tracker """
import numpy as np
from pythreshold.global_th.p_tile import p_tile_threshold
from image import filter_image, find_circles, segment_spin
import pickle

import cv2
from program import Program
from camera import Cameras
import matplotlib.pyplot as plt

def main():
  """ entry point  """
  test_calibrate() # uses photos

def test_calibrate():
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  chessboard_size = (8, 6)
  obj_points = []
  img_points_r = []
  img_points_l = []
  img_shape = None
  chess_img_r = None
  chess_img_l = None

  objp = np.zeros((np.prod(chessboard_size),3), dtype=np.float32)
  objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

  for i in range(0, 14):
    t = str(i)
    chessboard_r_found = False
    chessboard_l_found = False
    corners_r = None
    corners_l = None
    chess_img_r = cv2.imread('chessboard02/chessboard-R'+t+'.png', 0)
    chess_img_l = cv2.imread('chessboard02/chessboard-L'+t+'.png', 0)
    img_shape = chess_img_r.shape[:2]
    try:
      chessboard_r_found, corners_r = cv2.findChessboardCorners(chess_img_r, chessboard_size, None)
      chessboard_l_found, corners_l = cv2.findChessboardCorners(chess_img_l, chessboard_size, None)
    except Exception as ex:
      print('error finding chessboard {0}'.format(t))

    if chessboard_r_found == True & chessboard_l_found == True:
      print('Found {0}'.format(t))
      cv2.cornerSubPix(chess_img_r, corners_r, (5, 5), (-1, -1), criteria)
      cv2.cornerSubPix(chess_img_l, corners_l, (5, 5), (-1, -1), criteria)
      obj_points.append(objp)
      img_points_r.append(corners_r)
      img_points_l.append(corners_l)

  # calibrate cameras
  ret_r, K_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(obj_points, img_points_r, chess_img_r.shape[::-1], None, None)
  ret_l, K_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(obj_points, img_points_l, chess_img_l.shape[::-1], None, None)

  new_camera_matrix_r, roi_r = cv2.getOptimalNewCameraMatrix(K_r, dist_r, img_shape, 1, img_shape)
  new_camera_matrix_l, roi_l = cv2.getOptimalNewCameraMatrix(K_l, dist_l, img_shape, 1, img_shape)

  cameras = Cameras()
  while True:
    frame_r, frame_l = cameras.grab_one()

    res = np.hstack((frame_r, frame_l))
    cv2.imshow('camera', res)

    undistort_r = cv2.undistort(frame_r, K_r, dist_r, None, new_camera_matrix_r)
    undistort_l = cv2.undistort(frame_l, K_l, dist_l, None, new_camera_matrix_l)

    # undistort_r = downsample_image(undistort_r, 3)
    # undistort_l = downsample_image(undistort_l, 3)

    #Set disparity parameters
    #Note: disparity range is tuned according to specific parameters obtained through trial and error. 
    win_size = 5
    min_disp = -1
    max_disp = 31 #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16

    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=5,
                                  uniquenessRatio=5,
                                  speckleWindowSize=5,
                                  speckleRange=5,
                                  disp12MaxDiff=1,
                                  P1=8*3*win_size**2,#8*3*win_size**2,
                                  P2=32*3*win_size**2) #32*3*win_size**2)

    disparity_map = stereo.compute(undistort_r, undistort_l)

    plt.imshow(disparity_map,'gray')
    plt.show()

    k = cv2.waitKey(0) & 0xFF
    if k == 27:
      break

#Function that Downsamples image x number (reduce_factor) of times.
def downsample_image(image, reduce_factor):
  for i in range(0,reduce_factor):
    #Check if image is color or grayscale
    if len(image.shape) > 2:
      row,col = image.shape[:2]
    else:
      row,col = image.shape

    image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
  return image

if __name__ == '__main__':
  main()
