""" entry point to tracker """
import numpy as np
import pickle
from pickle import HIGHEST_PROTOCOL

import cv2
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
  corners_l = None
  corners_r = None

  objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
  objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

  for i in range(1, 113):
    t = str(i)
    chessboard_l_found = False
    chessboard_r_found = False
    corners_l = None
    corners_r = None
    chess_img_l = cv2.imread('chessboard02/chessboard-L'+t+'.png', 0)
    chess_img_r = cv2.imread('chessboard02/chessboard-R'+t+'.png', 0)
    img_shape = chess_img_r.shape[:2]
    try:
      chessboard_l_found, corners_l = cv2.findChessboardCorners(chess_img_l, chessboard_size, None)
      chessboard_r_found, corners_r = cv2.findChessboardCorners(chess_img_r, chessboard_size, None)
    except Exception as ex:
      print('error finding chessboard {0}'.format(t))

    if chessboard_l_found == True & chessboard_r_found == True:
      print('Found {0}'.format(t))
      cv2.cornerSubPix(chess_img_l, corners_l, (5, 5), (-1, -1), criteria)
      cv2.cornerSubPix(chess_img_r, corners_r, (5, 5), (-1, -1), criteria)
      obj_points.append(objp)
      img_points_l.append(corners_l)
      img_points_r.append(corners_r)

  # calibrate cameras
  ret_l, K_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(obj_points, img_points_l, chess_img_l.shape[::-1], None, None)
  ret_r, K_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(obj_points, img_points_r, chess_img_r.shape[::-1], None, None)

  new_camera_matrix_l, roi_l = cv2.getOptimalNewCameraMatrix(K_l, dist_l, img_shape, 1, img_shape)
  new_camera_matrix_r, roi_r = cv2.getOptimalNewCameraMatrix(K_r, dist_r, img_shape, 1, img_shape)

  left_camera = (new_camera_matrix_l, K_l, dist_l)
  with open('camera_matrix_l.pickle', 'wb') as handle:
    pickle.dump(left_camera, handle, protocol=HIGHEST_PROTOCOL)

  right_camera = (new_camera_matrix_r, K_r, dist_r)
  with open('camera_matrix_r.pickle', 'wb') as handle:
    pickle.dump(right_camera, handle, protocol=HIGHEST_PROTOCOL)

  # stereo calibrate
  stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                          cv2.TERM_CRITERIA_EPS, 100, 1e-5)

  retS, matrix_l, dist_coef_l, matrix_r, dist_coef_r, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_l, img_points_r, new_camera_matrix_l, dist_l, new_camera_matrix_r, dist_r, img_shape, criteria=stereocalib_criteria, flags=cv2.CALIB_FIX_INTRINSIC)
  rot_l, rot_r, proj_l, proj_r, Q, roi_l, roi_r = cv2.stereoRectify(matrix_l, dist_coef_l, matrix_r, dist_coef_r, img_shape, R, T)

  print('retS.stereoCalibrate - ', retS, ' retL - ', ret_l, ' retR - ', ret_r)

  left_map = cv2.initUndistortRectifyMap(matrix_l, dist_coef_l, rot_l, new_camera_matrix_l, img_shape, cv2.CV_16SC2)
  right_map = cv2.initUndistortRectifyMap(matrix_r, dist_coef_r, rot_r, new_camera_matrix_r, img_shape, cv2.CV_16SC2)

  stereo_map = (left_map, right_map)
  with open('stereo_map.pickle', 'wb') as handle:
    pickle.dump(stereo_map, handle, protocol=HIGHEST_PROTOCOL)

  window_size = 5
  min_disp = 15
  num_disp = 191-min_disp
  blockSize = window_size
  uniquenessRatio = 1
  speckleRange = 3
  speckleWindowSize = 3
  disp12MaxDiff = 200
  stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = uniquenessRatio,
    speckleRange = speckleRange,
    speckleWindowSize = speckleWindowSize,
    disp12MaxDiff = disp12MaxDiff,
    P1=8*3*window_size**2,#8*3*win_size**2,
    P2=32*3*window_size**2
  )

  cameras = Cameras()
  while True:
    frame_l, frame_r = cameras.grab_one()

    res = np.hstack((frame_l, frame_r))
    cv2.imshow('original', res)

    print('roi_l:', roi_l, ' - roi_r:', roi_r)
    ret = cv2.getValidDisparityROI(roi_l, roi_r, 1, 10, 5)
    print(ret)
    undistort_l = cv2.undistort(frame_l, K_l, dist_l, None, new_camera_matrix_l)
    # x,y,w,h = ret
    # undistort_l = undistort_l[y:y+h, x:x+w]
    undistort_r = cv2.undistort(frame_r, K_r, dist_r, None, new_camera_matrix_r)
    # x,y,w,h = ret
    # undistort_r = undistort_r[y:y+h, x:x+w]

    # disp = stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0
    # cv2.imshow('disparity', (disp-min_disp)/num_disp)

    # res = np.hstack((undistort_l, undistort_r))
    cv2.imshow('undistort_l', undistort_l)
    cv2.imshow('undistort_r', undistort_r)

    left_rectified = cv2.remap(undistort_l, left_map[0], left_map[1], interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    right_rectified = cv2.remap(undistort_r, right_map[0], right_map[1], interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow('left_rectified', left_rectified)
    cv2.imshow('right_rectified', right_rectified)

    #Generate  point cloud. 
    # h, w = undistort_r.shape[:2]
    # focal_length = 6
    # Q2 = np.float32([[1, 0, 0, 0],
    #                  [0, -1, 0, 0],
    #                  [0, 0, focal_length*0.05, 0], #Focal length multiplication obtained experimentally.
    #                  [0, 0, 0, 1]])

    # points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
    # mask_map = disparity_map > disparity_map.min()

    k = cv2.waitKey(0) & 0xFF
    if k == 115:
      cv2.imwrite('chessboard02/img-L.png', frame_l)
      cv2.imwrite('chessboard02/img-R.png', frame_r) # Save the image in the file where this Programm is located
    if k == 27:
      break

if __name__ == '__main__':
  main()
