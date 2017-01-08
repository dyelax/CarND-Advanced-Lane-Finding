import cv2
from glob import glob
from os.path import join, exists
import pickle

import constants as c

##
# I/O
##

def read_input(path):
    """
    Reads images from an input path into a numpy array. Paths can either be .jpg for single images
    or .mp4 for videos.

    :param path: The path to read.

    :return: A numpy array of images.
    """
    # TODO: Check img vs video.

    # TODO: Parse path.

    # TODO: Return images.
    pass


##
# Calibration / Undistortion
##

def calibrate_camera():
    """
    Calibrate the camera with the given calibration images.

    :return: A tuple (camera matrix, distortion coefficients)
    """
    # Check if camera has been calibrated previously.
    if exists(c.CALIBRATION_DATA):
        # TODO: Return pickled data.
        pass

    # If not, calibrate the camera.
    # TODO: Get calibration images.

    # TODO: For every calibration image, get object points and image points by finding chessboard corners.

    # TODO: Calculate camera matrix and distortion coefficients and return.
    pass

def undistort_imgs(imgs, camera_mat, dist_coeffs):
    """
    Undistorts distorted images.

    :param imgs: The distorted images.
    :param camera_mat: The camera matrix calculated from calibration.
    :param dist_coeffs: The distortion coefficients calculated from calibration.

    :return:
    """

##
# Testing
##

def display_image(img):
    """
    Displays an image and waits for a keystroke to dismiss and continue.

    :param img: The image to display
    """
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

