""" entry point to tracker """
import numpy as np
import pickle
from pickle import HIGHEST_PROTOCOL

import cv2
from camera import Cameras
import matplotlib.pyplot as plt

DIR_NAME = 'chessboard03'
CHESSBOARD_SIZE = (9, 6)

def main():
  """ entry point  """
  test_calibrate() # uses photos

def test_calibrate():
  flags = (cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_MARKER) # for findChessboardCornersSB
  obj_points = []
  img_points_r = []
  img_points_l = []
  img_shape = None
  chess_img_r = None
  chess_img_l = None
  corners_l = None
  corners_r = None

  objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), dtype=np.float32)
  objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

  for i in range(1, 113):
    t = str(i)
    chessboard_l_found = False
    chessboard_r_found = False
    corners_l = None
    corners_r = None
    chess_img_l = cv2.imread('{0}/chessboard-L{1}.png'.format(DIR_NAME, t), 0)
    chess_img_r = cv2.imread('{0}/chessboard-R{1}.png'.format(DIR_NAME, t), 0)
    img_shape = chess_img_r.shape[:2]
    try:
      chessboard_l_found, corners_l = cv2.findChessboardCornersSB(chess_img_l, CHESSBOARD_SIZE, flags=flags)
      chessboard_r_found, corners_r = cv2.findChessboardCornersSB(chess_img_r, CHESSBOARD_SIZE, flags=flags)
    except Exception as ex:
      print('error finding chessboard {0}'.format(t))

    if chessboard_l_found == True & chessboard_r_found == True:
      print('Found {0}'.format(t))
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
  print('R: ', R)
  print('T:', T)

  left_map = cv2.initUndistortRectifyMap(matrix_l, dist_coef_l, rot_l, proj_l, img_shape, cv2.CV_16SC2)
  right_map = cv2.initUndistortRectifyMap(matrix_r, dist_coef_r, rot_r, proj_r, img_shape, cv2.CV_16SC2)

  stereo_map = (left_map, right_map)
  with open('stereo_map.pickle', 'wb') as handle:
    pickle.dump(stereo_map, handle, protocol=HIGHEST_PROTOCOL)

  cameras = Cameras()
  while True:
    frame_l, frame_r = cameras.grab_one()

    res = np.hstack((frame_l, frame_r))
    cv2.imshow('original', res)

    print('roi_l:', roi_l, ' - roi_r:', roi_r)
    ret = cv2.getValidDisparityROI(roi_l, roi_r, 1, 10, 5)
    print(ret)

    left_rectified = cv2.remap(frame_l, left_map[0], left_map[1], interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    right_rectified = cv2.remap(frame_r, right_map[0], right_map[1], interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow('left_rectified', left_rectified)
    cv2.imshow('right_rectified', right_rectified)

    img_to_use_l = left_rectified
    img_to_use_r = right_rectified

    chessboard_l_found, corners_l = cv2.findChessboardCorners(img_to_use_l, CHESSBOARD_SIZE, None)
    chessboard_r_found, corners_r = cv2.findChessboardCorners(img_to_use_r, CHESSBOARD_SIZE, None)
    print(chessboard_l_found, chessboard_r_found)
    if chessboard_l_found == True & chessboard_r_found == True:
      print('point_l: ', corners_l[0][0][0])
      print('point_r: ', corners_r[0][0][0])
      disparity_maybe = corners_l[0][0][0] - corners_r[0][0][0]
      disp_may = (corners_l[0][0][0], corners_l[0][0][1], disparity_maybe)
      np_disp = np.array([[[corners_l[0][0][0], corners_l[0][0][1], disparity_maybe]]], dtype=np.float32)
      new_coord = cv2.perspectiveTransform(np_disp, Q)
      print('new_coord: ', new_coord)

    k = cv2.waitKey(0) & 0xFF
    if k == 115:
      cv2.imwrite('{0}/img-L.png'.format(DIR_NAME), frame_l)
      cv2.imwrite('{0}/img-R.png'.format(DIR_NAME), frame_r)
    if k == 27:
      break

if __name__ == '__main__':
  main()
