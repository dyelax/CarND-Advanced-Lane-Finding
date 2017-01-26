from os import makedirs
from os.path import join, exists
import numpy as np


def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not exists(directory):
        makedirs(directory)
    return directory

DATA_DIR = '../data/'
CALIBRATION_DIR = join(DATA_DIR, 'camera_cal/')
CALIBRATION_DATA_PATH = join(CALIBRATION_DIR, 'calibration_data.p')
TEST_DIR = join(DATA_DIR, 'test_images/')

SAVE_DIR = '../output_images/'


# Points picked from an image with straight lane lines.
SRC = np.float32([
    (257, 685),
    (1050, 685),
    (583, 460),
    (702, 460)
])
# Mapping from those points to a rectangle for a birdseye view.
DST = np.float32([
    (200, 720),
    (1080, 720),
    (200, 0),
    (1080, 0)
])

# The pixel width of lines to assume in images.
LINE_WIDTH = 200
# The pixel width of lines from the center point. Used for searching for lines around a peak.
LINE_RADIUS = LINE_WIDTH / 2

IMG_HEIGHT = 720
IMG_WIDTH = 1280

# The number of most recent elements from the fit history to consider when looking for new lines.
RELEVANT_HIST = 5

# meters per pixel in y dimension
MPP_Y = 30. / 720
# meters per pixel in x dimension
MPP_X = 3.7 / 880