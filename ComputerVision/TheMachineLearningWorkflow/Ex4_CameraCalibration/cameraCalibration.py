import sys
import numpy as np
import glob
import cv2
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def cal_undistort(img, objpoints, imgpoints):
    """
    Fix distortion the img
    :param img: distorted image
    :param objpoints: The expected points
    :param imgpoints: The distorted points
    :return: cv2 sucess function, camera matrix, distortion coefficient, rotation, translation (of world pose),
    undistorted image.
    """
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return ret, mtx, dist, rvecs, tvecs, undist


def camera_calibration(cb_images_dir_path: str, cb_images_file_extension: str, cb_images_name: str, cb_dim: tuple):
    """
    Compute corners in ches board
    Note: cb is a shortcut of chess board
    :param cb_images_dir_path: The directory where the chess board images are
    :param cb_images_file_extension: such as JPEG, PNG etc.
    :param cb_images_name: chess board images name. Should be of the form "name" + index for example
    "calibration1" so this arg should receive only the "calibration" name
    :param cb_dim: tuple of the form (height, width)
    :return: images, corners coordinates and the return values of this comutation
    """
    # Read in and make a list of images calibration
    images = glob.glob(cb_images_dir_path + "/" + cb_images_name + "*." + cb_images_file_extension)

    img_shape = images[0].shape

    # Arrays to store object points and image points from all the images
    obj_points = []
    img_points = []
    cb_height, checkers_board_width = cb_dim[0], cb_dim[1]

    # prepare object points (0, 0, 0) , (1, 0, 0)
    objp = np.zeroes((checkers_board_width * cb_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkers_board_width, 0:cb_height].T.reshape(-1, 2)  # x, y coordinates

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the checkers board corners
        ret, corners = cv2.findChessboardCorners(gray, (checkers_board_width, cb_height), None)

        # If objects are found, add object points and image points
        if ret:
            img_points.append(corners)
            obj_points.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape[1::-1], None, None)
    if ret:
        return ret, mtx, dist, rvecs, tvecs

    raise ValueError('Open cv "calibrateCamera" function failed')


def display_checkers_board_corners(images_lst, checkers_board_dim, corners_lst, ret_lst, saving_path=None):
    """
    Draws a chess board with the corners outputted from cv2
    :param images_lst:
    :param checkers_board_dim:
    :param corners_lst:
    :param ret_lst:
    :param saving_path:
    """
    checkers_board_height, checkers_board_width = checkers_board_dim[0], checkers_board_dim[1]
    for img, corners, ret in zip(images_lst, corners_lst, ret_lst):
        img_with_pts = cv2.drawChessboardCorners(img, (checkers_board_width, checkers_board_height), corners, ret)
        if saving_path is not None:
            img_with_pts.imwrite(saving_path)
        else:
            plt.imshow(img_with_pts)


if __name__ == "__main__":
    camera_calibration(sys.argv[1], sys.argv[2], sys.argv[3],
                       tuple(map(int, sys.argv[4].split(', '))))

