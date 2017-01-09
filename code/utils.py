import numpy as np
import cv2
from glob import glob
from os.path import join, exists, splitext
import pickle

import constants as c
from line import Line


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
    ext = splitext(path)[1]
    assert ext == '.jpg' or ext == '.mp4', 'The input file must be a .jpg or .mp4.'

    if ext == '.jpg':
        # Input is a single image.
        img = cv2.imread(path)
        frames = np.array([img])  # turn into a 4D array so all functions can apply to images and video.
    else:
        # Input is a video.
        vidcap = cv2.VideoCapture(path)

        # Get video properties
        num_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Load frames
        frames = np.empty([num_frames, height, width, 3])
        while vidcap.isOpened():
            frame_num = vidcap.get(cv2.CAP_PROP_POS_FRAMES)

            ret, frame = vidcap.read()

            if ret:
                frames[frame_num] = frame

        vidcap.release()

    return frames


def save(imgs, path):
    """
    Saves imgs to file. Paths can either be .jpg for single images or .mp4 for videos.

    :param imgs: The frames to save. A single image for .jpgs, or multiple frames for .mp4s.
    :param path: The path to which the image / video will be saved.
    """
    ext = splitext(path)[1]
    assert ext == '.jpg' or ext == '.mp4', 'The output file must be a .jpg or .mp4.'

    if ext == '.jpg':
        # Output is a single image.
        cv2.imwrite(path, imgs[0])
    else:
        # Output is a video.
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        # Params: path, format, fps, frame size.
        writer = cv2.VideoWriter(path, fourcc, 30.0, imgs.shape[1:3])

        for img in imgs:
            writer.write(img)

        writer.release()

##
# Calibration / Image Processing
##

def calibrate_camera():
    """
    Calibrate the camera with the given calibration images.

    :return: A tuple (camera matrix, distortion coefficients)
    """
    # Check if camera has been calibrated previously.
    if exists(c.CALIBRATION_DATA_PATH):
        # Return pickled calibration data.
        pickle_dict = pickle.load(open(c.CALIBRATION_DATA_PATH, "rb"))
        camera_mat = pickle_dict["camera_mat"]
        dist_coeffs = pickle_dict["dist_coeffs"]

        print 'Calibration data loaded!'

        return camera_mat, dist_coeffs

    # If not, calibrate the camera.
    print 'Calibrating camera...'

    # For every calibration image, get object points and image points by finding chessboard corners.
    obj_points = []  # 3D points in real world space.
    img_points = []  # 2D points in image space.

    # Prepare constant object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0).
    obj_points_const = np.zeros((6 * 8, 3), np.float32)
    obj_points_const[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    filenames = glob(join(c.CALIBRATION_DIR, '*.jpg'))
    gray_shape = None
    for path in filenames:
        img = cv2.imread(path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)

        if ret:
            obj_points.append(obj_points_const)
            img_points.append(corners)

    # Calculate camera matrix and distortion coefficients and return.
    ret, camera_mat, dist_coeffs, _, _ = cv2.calibrateCamera(obj_points, img_points, gray_shape, None, None)

    assert ret, 'CALIBRATION FAILED'  # Make sure calibration didn't fail.

    # Save calibration data
    pickle_dict = {'camera_mat': camera_mat, 'dist_coeffs': dist_coeffs}
    pickle.dump(pickle_dict, open(c.CALIBRATION_DATA_PATH, 'wb'))

    return camera_mat, dist_coeffs


def undistort_imgs(imgs, camera_mat, dist_coeffs):
    """
    Undistorts distorted images.

    :param imgs: The distorted images.
    :param camera_mat: The camera matrix calculated from calibration.
    :param dist_coeffs: The distortion coefficients calculated from calibration.

    :return:
    """
    imgs_undist = np.empty_like(imgs)
    for i, img in enumerate(imgs):
        imgs_undist[i] = cv2.undistort(img, camera_mat, dist_coeffs, None, camera_mat)

    return imgs_undist


##
# Masking
##

def get_s_channel(img):
    """
    Returns the saturation channel of a BGR image.

    :param img: The image from which to extract the saturation channel.

    :return: The saturation channel of img.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    return s_channel


def quad_mask(img, bl=(), br=(), tl=(), tr=()):
    """
    Creates a binary quadrilateral (usually trapezoidal) mask for the image to isolate the area where lane lines are.

    :param img: The image for which to create a mask.
    :param bl: The bottom-left vertex of the quadrilateral.
    :param br: The bottom-right vertex of the quadrilateral.
    :param tl: The top-left vertex of the quadrilateral.
    :param tr: The top-right vertex of the quadrilateral.

    :return: A binary mask with all pixels from img inside the quadrilateral masked.
    """
    fit_left = np.polyfit((bl[0], tl[0]), (bl[1], tl[1]), 1)
    fit_right = np.polyfit((br[0], tr[0]), (br[1], tr[1]), 1)
    fit_bottom = np.polyfit((bl[0], br[0]), (bl[1], br[1]), 1)
    fit_top = np.polyfit((tl[0], tr[0]), (tl[1], tr[1]), 1)

    # Find the region inside the lines
    xs, ys = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    mask = (xs > (ys * fit_left[0] + fit_left[1])) & \
           (xs < (ys * fit_right[0] + fit_right[1])) & \
           (ys > (xs * fit_top[0] + fit_top[1])) & \
           (ys < (xs * fit_bottom[0] + fit_bottom[1]))

    return mask


def color_mask(img, thresh=(170, 255)):
    """
    Creates a binary mask for an image based on threshold values of the saturation channel.

    :param img: The image for which to create a mask.
    :param thresh: The threshold values between which to mask.

    :return: A binary mask with all pixels from img with saturation level inside the threshold masked.
    """
    s_channel = get_s_channel(img)

    mask = (s_channel > thresh[0]) & (s_channel < thresh[1])

    return mask


def grad_mask(img, thresh=(20, 100)):
    """
    Creates a binary mask for an image based on threshold values of the x gradient (detecting vertical lines).

    :param img: The image for which to create a mask.
    :param thresh: The threshold values between which to mask.

    :return: A binary mask with all pixels from img with saturation level inside the threshold masked.
    """
    s_channel = get_s_channel(img)

    # Take the gradient in the x direction.
    sx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)
    sx_abs = np.absolute(sx)
    sx_scaled = np.uint8(255 * sx_abs / np.max(sx_abs))

    mask = (sx_scaled > thresh[0]) & (sx_scaled < thresh[1])

    return mask


def get_masks(imgs):
    """
    Creates binary masks for images based on color, x-gradient and position.

    :param imgs: The images for which to create masks.

    :return: A binary mask for each image in imgs where the lane lines are masked.
    """
    # Each mask will be a single channel, so ignore depth of input images
    masks = np.empty(imgs.shape[:-1])

    for i, img, in enumerate(imgs):
        masks[i] = np.uint8(quad_mask(img) & color_mask(img) & grad_mask(img))

    return masks


def birdseye(imgs):
    """
    Shift the perspective of an image to a birdseye view.

    :param imgs: The images to be transformed.

    :return: The images, perspective shifted to a birdseye view.
    """
    imgs_birdseye = np.empty_like(imgs)

    # TODO: Get source and destination rects.
    trans_mat = cv2.getPerspectiveTransform(dst, src)
    for i, img in enumerate(imgs):
        imgs_birdseye[i] = cv2.warpPerspective(img, trans_mat, img.shape, flags=cv2.INTER_LINEAR)

    return imgs_birdseye


##
# Find lines
##

def fit_line(points, meter_space=True):
    """
    Fit a second-order polynomial to the given points.

    :param points: The points on which to fit the line.
    :param meter_space: Whether to calculate the fit in meter-space (True) or pixel-space (False).

    :return: The coefficients of the fitted polynomial.
    """
    # TODO: Define conversions in x and y from pixels space to meters
    # THESE MAY NOT BE CORRECT
    mpp_y = 30 / 720  # meters per pixel in y dimension
    mpp_x = 3.7 / 700  # meteres per pixel in x dimension

    # Determine whether to fit the line in meter or pixel space.
    ymult = mpp_y if meter_space else 1
    xmult = mpp_x if meter_space else 1

    # noinspection PyTypeChecker
    fit = np.polyfit(points[1] * ymult, points[0] * xmult, 2)

    return fit

def get_curvature_radius(line, y):
    """
    Get the curvature radius of the given lane line at the given y coordinate.

    :param line: The line of which to calculate the curvature radius. NOTE: Must be in real-world coordinates to
                 accurately calculate road curvature.

    :return: The curvature radius of line at y.
    """
    A, B, C = line
    return np.power(1 + np.square(2 * A * y + B), 3 / 2) / np.abs(2 * A)


def lines_good(l, r):
    """
    Determines if the two lane lines make sense.

    :param l: The polynomial fit for the left line.
    :param r: The polynomial fit for the right line.

    :return: A boolean, whether the two lane lines make sense.
    """
    # TODO: Check similar curvature
    # TODO: Check parallel
    # TODO: Check correct width apart
    pass


def find_lines(masks):
    """
    Get lane line equations from image masks.

    :param masks: Binary mask images of the lane lines.

    :return: A tuple, (left line, right line).
    """
    lines = []

    # Create line objects.
    l = Line()
    r = Line()

    for mask in masks:
        # TODO: Look for points based on past lines (within x threshold of prev values)
        # TODO: Fit lines to those points

        if lines_good(l, r):
            lines.append((l, r))
        else:
            # TODO: Use histogram to get line points
            # TODO: Fit lines to those points
            pass

    return lines


def draw_lane(imgs, l_r_c):
    """
    Superimposes the lane on the original images.

    :param imgs: The original images.
    :param l_r_c: A list containing a tuple for each image, (left line, right line, curvature radius).

    :return: Images consisting of the lane prediction superimposed on the original street image.
    """
    imgs_superimposed = np.empty_like(imgs)

    for i, img in enumerate(imgs):
        # TODO: Get polygon for lane
        # TODO: Superimpose polygon onto original image
        pass

    return imgs_superimposed

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

