import numpy as np
import cv2
import skvideo.io
from glob import glob
from os.path import join, exists, splitext
import pickle
import matplotlib.pyplot as plt

import constants as c
from lines import Lines


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
        # turn into a 4D array so all functions can apply to images and video.
        frames = np.array([img])
    else:
        # Input is a video.
        vidcap = cv2.VideoCapture(path)

        # Get video properties
        # num_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        # width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Load frames
        frames_list = []
        while vidcap.isOpened():
            ret, frame = vidcap.read()

            if ret:
                frames_list.append(frame)
            else:
                break

        vidcap.release()

        frames = np.array(frames_list)

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
        vid_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        skvideo.io.vwrite(path, vid_frames)


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

    # Prepare constant object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0).
    obj_points_const = np.zeros((6 * 9, 3), np.float32)
    obj_points_const[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    filenames = glob(join(c.CALIBRATION_DIR, '*.jpg'))
    gray_shape = None
    for path in filenames:
        img = cv2.imread(path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            obj_points.append(obj_points_const)
            img_points.append(corners)

    # Calculate camera matrix and distortion coefficients and return.
    ret, camera_mat, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, gray_shape, None, None)

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

def get_gray(img):
    """
    Converts a BGR image to grayscale.

    :param img: The image to be converted to grayscale

    :return: Img in grayscale.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_hls(img):
    """
    Returns the hue, light and saturation channels of a BGR image.

    :param img: The image from which to extract the hsl channels.

    :return: Img in HLS space.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    return hls


def quad_mask(img):
    """
    Creates a binary quadrilateral (usually trapezoidal) mask for the image to isolate the area
    where lane lines are.

    :param img: The image for which to create a mask.

    :return: A binary mask with all pixels from img inside the quadrilateral masked.
    """
    height, width = img.shape[:2]
    bl = (width / 2 - 450, height - 20)  # The bottom-left vertex of the quadrilateral.
    br = (width / 2 + 650, height - 20)  # The bottom-right vertex of the quadrilateral.
    tl = (width / 2 - 50, height / 2 + 85)  # The top-left vertex of the quadrilateral.
    tr = (width / 2 + 80, height / 2 + 85)  # The top-right vertex of the quadrilateral.

    fit_left = np.polyfit((bl[0], tl[0]), (bl[1], tl[1]), 1)
    fit_right = np.polyfit((br[0], tr[0]), (br[1], tr[1]), 1)
    fit_bottom = np.polyfit((bl[0], br[0]), (bl[1], br[1]), 1)
    fit_top = np.polyfit((tl[0], tr[0]), (tl[1], tr[1]), 1)

    # Find the region inside the lines
    xs, ys = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    mask = (ys > (xs * fit_left[0] + fit_left[1])) & \
           (ys > (xs * fit_right[0] + fit_right[1])) & \
           (ys > (xs * fit_top[0] + fit_top[1])) & \
           (ys < (xs * fit_bottom[0] + fit_bottom[1]))

    return mask

def tri_anti_mask(img):
    """
    Creates a binary quadrilateral (usually trapezoidal) mask for the image to isolate the area
    where lane lines are.

    :param img: The image for which to create a mask.

    :return: A binary mask with all pixels from img inside the quadrilateral masked.
    """
    height, width = img.shape[:2]
    t = (width / 2, height / 2 + 120)
    bl = (width / 2 - 300, height - 20)
    br = (width / 2 + 420, height - 20)

    fit_left = np.polyfit((bl[0], t[0]), (bl[1], t[1]), 1)
    fit_right = np.polyfit((br[0], t[0]), (br[1], t[1]), 1)
    fit_bottom = np.polyfit((bl[0], br[0]), (bl[1], br[1]), 1)

    # Find the region outside the lines
    xs, ys = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    mask = (ys < (xs * fit_left[0] + fit_left[1])) | \
           (ys < (xs * fit_right[0] + fit_right[1])) | \
           (ys > (xs * fit_bottom[0] + fit_bottom[1]))

    return mask


def color_mask(img, thresh_s=(100, 255), thresh_h=(15, 70), thresh_gray=(190,255)):
    """
    Creates a binary mask for an image based on threshold values of the saturation channel.

    :param img: The image for which to create a mask.
    :param thresh_s: The saturation threshold values between which to mask.
    :param thresh_h: The hue threshold values between which to mask.
    :param thresh_gray: The grayscale threshold values between which to mask.

    :return: A binary mask with all pixels from img with hue and saturation levels inside the
             thresholds masked.
    """
    hls = get_hls(img)
    h_channel = hls[:, :, 0]
    s_channel = hls[:, :, 2]

    gray = get_gray(img)

    s_mask = ((s_channel > thresh_s[0]) & (s_channel < thresh_s[1]))
    h_mask = ((h_channel > thresh_h[0]) & (h_channel < thresh_h[1]))
    gray_mask = ((gray > thresh_gray[0]) & (gray < thresh_gray[1]))

    mask = (s_mask | h_mask | gray_mask)

    return mask


def sobel(img, x=True, y=False, scaled=True, ksize=3):
    """
    Calculate the scaled Sobel operation on an image.

    :param img: The image on which to perform Sobel.
    :param x: Whether to perform Sobel on the x axis.
    :param y: Whether to perform Sobel on the y axis.
    :param scaled: Whether to scale the result to the range [0, 255] (True) or keep in the range
                   [0, 1] (False).
    :param ksize: The kernel size for the Sobel operation.

    :return: The result of the Sobel operation applied to img on the specified axis.
    """
    s = cv2.Sobel(img, cv2.CV_64F, x, y, ksize=ksize)
    s_abs = np.abs(s)
    scale = 255 if scaled else 1
    s_scaled = np.uint8(scale * s_abs / np.max(s_abs))

    return s_scaled


def grad_mask(img, thresh_s=(30, 255), thresh_gray=(50, 255)):
    """
    Creates a binary mask for an image based on threshold values of the x gradient (detecting
    vertical lines).

    :param img: The image for which to create a mask.
    :param thresh_s: The saturation gradient threshold values between which to mask.
    :param thresh_gray: The grayscale gradient threshold values between which to mask.

    :return: A binary mask with all pixels from img with saturation level inside the threshold
             masked.
    """
    s_channel = get_hls(img)[:, :, 2]
    gray = get_gray(img)

    # Take the gradient in the x direction.
    sobel_s = sobel(s_channel, ksize=7)
    sobel_gray = sobel(gray, ksize=7)

    s_mask = ((sobel_s > thresh_s[0]) & (sobel_s < thresh_s[1]))
    gray_mask = ((sobel_gray > thresh_gray[0]) & (sobel_gray < thresh_gray[1]))

    mask = (s_mask | gray_mask)

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
        masks[i] = np.uint8((quad_mask(img) & tri_anti_mask(img)) &
                            (color_mask(img) & grad_mask(img)))

    return masks


def birdseye(imgs, inverse=False):
    """
    Shift the perspective of an image to or from a birdseye view.

    :param imgs: The images to be transformed.
    :param inverse: Whether to do an inverse transformation (ie. birdseye to normal).
                    Default = False.

    :return: The perspective shifted images.
    """
    imgs_transformed = np.empty_like(imgs)
    height, width = imgs.shape[1:3]

    if inverse:
        src = c.DST
        dst = c.SRC
    else:
        src = c.SRC
        dst = c.DST

    trans_mat = cv2.getPerspectiveTransform(src, dst)
    for i, img in enumerate(imgs):
        transformed = cv2.warpPerspective(img, trans_mat, (width, height), flags=cv2.INTER_LINEAR)
        imgs_transformed[i] = transformed

    return imgs_transformed


##
# Find lines
##

def visualize_lines(left_points, right_points, left_line, right_line):
    """
    Visualize lane line fits for given line points.

    :param left_points: Points in the left line.
    :param right_points: Points in the right line.
    :param left_line: The polynomial fit for the left line.
    :param right_line: The polynomial fit for the right line.
    """
    plt.gcf().clear()

    yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image

    left_fit_x = left_line[0] * yvals ** 2 + left_line[1] * yvals + left_line[2]
    right_fit_x = right_line[0] * yvals ** 2 + right_line[1] * yvals + right_line[2]

    plt.plot(left_points[0], left_points[1], 'o', color='blue')
    plt.plot(right_points[0], right_points[1], 'o', color='blue')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fit_x, yvals, color='green', linewidth=3)
    plt.plot(right_fit_x, yvals, color='red', linewidth=3)
    plt.gca().invert_yaxis()  # to visualize as we do the images

    plt.show()

def fit_line(points, meter_space=False):
    """
    Fit a second-order polynomial to the given points.

    :param points: The points on which to fit the line.
    :param meter_space: Whether to calculate the fit in meter-space (True) or pixel-space (False).

    :return: The coefficients of the fitted polynomial.
    """
    # Determine whether to fit the line in meter or pixel space.
    ymult = c.MPP_Y if meter_space else 1
    xmult = c.MPP_X if meter_space else 1

    # noinspection PyTypeChecker
    fit = np.polyfit(points[1] * ymult, points[0] * xmult, 2)

    return fit


def get_curvature_radius(line, y):
    """
    Get the curvature radius of the given lane line at the given y coordinate.

    :param line: The line of which to calculate the curvature radius. NOTE: Must be in real-world
                 coordinates to accurately calculate road curvature.

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
    # Check parallel
    # parallel_tolerance = 5e-4
    # parallel = (l[0] > r[0] - parallel_tolerance) and (l[0] < r[0] + parallel_tolerance)

    # Check parallel and correct width apart
    width_avg = 880  # taken from dst points
    width_tolerance = 150
    yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image
    left_fit_x = l[0] * yvals ** 2 + l[1] * yvals + l[2]
    right_fit_x = r[0] * yvals ** 2 + r[1] * yvals + r[2]

    correct_width = True
    for l_x, r_x in zip(left_fit_x, right_fit_x):
        if r_x - l_x < width_avg - width_tolerance or r_x - l_x > width_avg + width_tolerance:
            correct_width = False
            break

    return correct_width


def hist_search(img, hist_height=80):
    """
    Uses a sliding histogram to search for lane lines in a masked, perspective shifted image.

    :param img: The masked, perspective shifted image containing the lane lines.
    :param hist_height: The height of the sliding window of the histogram.

    :return: A tuple of tuples ((left xs, left ys), (right xs, right ys)) of the points in both lane
             lines.
    """
    left_xs = []
    left_ys = []
    right_xs = []
    right_ys = []

    height, width = img.shape[:2]
    for i in xrange(0, height, hist_height):
        # Get slice of the image.
        window = img[i:i + hist_height, :]

        # Calculate histogram on slice.
        hist = np.sum(window, axis=0)
        # arr2bar(hist)

        # Find 2 peaks.
        middle = width / 2
        left_peak = np.argmax(hist[:middle])
        # Add back in the shift to get the real index.
        right_peak = np.argmax(hist[middle:]) + middle

        # Get a range around the peaks.
        left_rect = window[:, left_peak - c.LINE_RADIUS:left_peak + c.LINE_RADIUS]
        right_rect = window[:, right_peak - c.LINE_RADIUS:right_peak + c.LINE_RADIUS]

        # Get coordinates of points in rects
        # Relative locations of the points in hist as (rows (y), cols (x))
        left_points_relative = np.where(left_rect)
        right_points_relative = np.where(right_rect)

        # Absolute locations in img as (x, y)
        left_points = (left_points_relative[1] + left_peak - c.LINE_RADIUS,
                       left_points_relative[0] + i)
        right_points = (right_points_relative[1] + right_peak - c.LINE_RADIUS,
                        right_points_relative[0] + i)

        # Append points
        left_xs = np.concatenate([left_xs, left_points[0]])
        left_ys = np.concatenate([left_ys, left_points[1]])
        right_xs = np.concatenate([right_xs, right_points[0]])
        right_ys = np.concatenate([right_ys, right_points[1]])

    return (left_xs, left_ys), (right_xs, right_ys)

# TEST
def hist_mask_test(img):
    l, r = hist_search(img)
    output = np.zeros(img.shape + (3,))

    output[:, :, 0] = img * 255
    for x, y in zip(l[0], l[1]):
        x, y = int(x), int(y)  # To suppress warnings about float indeces.
        output[y, x, 1] = 255
        output[y, x, 0] = 0
    for x, y in zip(r[0], r[1]):
        x, y = int(x), int(y)
        output[y, x, 2] = 255
        output[y, x, 0] = 0

    return [output]

def local_search(img, history):
    """
    Finds lane line points based on most recent line fits.

    :param img: The masked, perspective shifted image containing the lane lines.
    :param history: A list of tuples, (left, right), the line fits used for previous frames.

    :return: A tuple of tuples ((left xs, left ys), (right xs, right ys)) of the points in both lane
             lines.
    """
    # Radius around the last good line to search for new line pixels.
    search_radius = c.LINE_RADIUS + 10

    # Must exist because of check in find_lines()
    last_good_fit = [elt for elt in history[-c.RELEVANT_HIST:] if elt is not None][-1]
    left_line_prev, right_line_prev = last_good_fit

    yvals = np.linspace(0, 719, num=720)  # to cover same y-range as image
    left_fit_x = left_line_prev[0] * yvals ** 2 + left_line_prev[1] * yvals + left_line_prev[2]
    right_fit_x = right_line_prev[0] * yvals ** 2 + right_line_prev[1] * yvals + right_line_prev[2]

    left_xs = []
    left_ys = []
    right_xs = []
    right_ys = []
    for row_num, row in enumerate(img):
        left_center = int(left_fit_x[row_num])
        right_center = int(right_fit_x[row_num])

        # Get a range around the lines.
        left_rect = row[left_center - search_radius:left_center + search_radius]
        right_rect = row[right_center - search_radius:right_center + search_radius]

        # Get coordinates of points in rects
        # Relative x locations of the points in the row
        left_points_relative = np.where(left_rect)
        right_points_relative = np.where(right_rect)

        # Absolute locations in img as (x, y)
        left_points = (left_points_relative[0] + left_center - search_radius,
                       [row_num] * len(left_points_relative[0]))
        right_points = (right_points_relative[0] + right_center - search_radius,
                        [row_num] * len(right_points_relative[0]))

        # Append points
        left_xs = np.concatenate([left_xs, left_points[0]])
        left_ys = np.concatenate([left_ys, left_points[1]])
        right_xs = np.concatenate([right_xs, right_points[0]])
        right_ys = np.concatenate([right_ys, right_points[1]])

    return (left_xs, left_ys), (right_xs, right_ys)


def find_lines(masks):
    """
    Get lane line equations from image masks.

    :param masks: Binary mask images of the lane lines.

    :return: A tuple (lines, history), where lines is a list containing a tuple for each image,
             (left fit, right fit). and history is a list where each element is a tuple (left fit,
             right fit) for a previous frame, or None if no fit was found for a frame. (used to
             display when fallback lines were used).
    """
    lines = []

    # a list where each element is a tuple (left fit, right fit) for a previous frame, or None if no
    # fit was found for a frame.
    history = []
    for mask in masks:
        # Until the first good line is found, do naive histogram search.
        if len([elt for elt in history if elt is not None]) == 0:
            left_points, right_points = hist_search(mask)
            left_fit = fit_line(left_points)
            right_fit = fit_line(right_points)

            if lines_good(left_fit, right_fit):
                history.append((left_fit, right_fit))
            else:
                history.append(None)

            lines.append((left_fit, right_fit))

            continue

        # After, choose lines with the following priority:
        # 1. First, try get the lines from a local search based on past lines found. If those lines
        #    are good, use them.
        # 2. If good lines were found in the last c.RELEVANT_HIST frames, use the most recent good
        #    lines found.
        # 3. Do a naive histogram search. If those lines are good, use them.
        # 4. Otherwise, use the last good fit found, regardless of time.

        # If a new fit was found and it was good (as opposed to taking a previous fit). Used to
        # place tuples or nones in history.
        new_good_fit = True

        # If good lines were found in the last c.RELEVANT_HIST frames, try priorities 1 and 2:
        if len([elt for elt in history[-c.RELEVANT_HIST:] if elt is not None]) != 0:
            # Priority 1:
            left_points, right_points = local_search(mask, history)
            left_fit = fit_line(left_points)
            right_fit = fit_line(right_points)

            if not lines_good(left_fit, right_fit):
                # Priority 2:
                last_good = [elt for elt in history[-c.RELEVANT_HIST:] if elt is not None][-1]
                left_fit, right_fit = last_good
                new_good_fit = False
        else:
            # No good fits in relevant history
            # Priority 3:
            left_points, right_points = hist_search(mask)
            left_fit = fit_line(left_points)
            right_fit = fit_line(right_points)

            if not lines_good(left_fit, right_fit):
                # Priority 4:
                new_good_fit = False
                # Will never fail because of check at beginning of loop
                last_good = [elt for elt in history if elt is not None][-1]
                left_fit, right_fit = last_good

        # Update history
        if new_good_fit:
            history.append((left_fit, right_fit))
        else:
            history.append(None)

        lines.append((left_fit, right_fit))

    return lines, history


def draw_lane(imgs, lines, history):
    """
    Superimposes the lane on the original images.

    :param imgs: The original images.
    :param lines: A list containing a tuple for each image, (left fit, right fit)
    :param history: a list where each element is a tuple (left fit, right fit) for a previous frame,
                    or None if no fit was found for a frame. (used to display when fallback lines
                    were used).

    :return: Images consisting of the lane prediction superimposed on the original street image.
    """
    imgs_superimposed = np.empty_like(imgs)

    yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image
    for i, img in enumerate(imgs):
        left_line, right_line = lines[i]

        # Create an image to draw the lines on
        overlay_warped = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        left_fit_x = left_line[0] * yvals ** 2 + left_line[1] * yvals + left_line[2]
        right_fit_x = right_line[0] * yvals ** 2 + right_line[1] * yvals + right_line[2]

        pts_left = np.array([np.transpose(np.vstack([left_fit_x, yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay_warped, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space
        overlay = birdseye(np.array([overlay_warped]), inverse=True)[0]

        # Overlay curvature text
        height = img.shape[0]
        left_fit_m = fit_line((left_fit_x, yvals), meter_space=True)
        right_fit_m = fit_line((right_fit_x, yvals), meter_space=True)
        left_curvature = get_curvature_radius(left_fit_m, height)
        right_curvature = get_curvature_radius(right_fit_m, height)
        curvature = (left_curvature + right_curvature) / 2

        text_color = (255, 255, 255)
        if history[i] is None:
            text_color = (0, 0, 255)

        cv2.putText(img, "Curvature Radius: " + str(curvature) + 'm', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # TODO: overlay distance from lane center
        # Negative distances represent left of center
        frame_center_m = (c.IMG_WIDTH / 2) * c.MPP_X
        left_bottom = left_fit_x[-1]
        right_bottom = right_fit_x[-1]

        dist_from_left = frame_center_m - left_bottom
        dist_from_right = right_bottom - frame_center_m
        dist_from_center = dist_from_left - dist_from_right

        cv2.putText(img, "Distance from Center: " + str(dist_from_center) + 'm', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)


        # Combine the result with the original image
        imgs_superimposed[i] = cv2.addWeighted(img, 1, overlay, 0.3, 0)

    return imgs_superimposed


##
# Testing
##

def arr2bar(arr):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(arr)), arr, 1)
    plt.show()


def display_images(imgs):
    """
    Displays an image and waits for a keystroke to dismiss and continue.

    :param imgs: The images to display
    """
    for img in imgs:
        # Conversion for masks
        if img.dtype == bool:
            img = np.uint8(img) * 255

        cv2.imshow('image', img)
        cv2.moveWindow('image', 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
