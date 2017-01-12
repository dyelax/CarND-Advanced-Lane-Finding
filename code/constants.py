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
    (215, 715),
    (1120, 715),
    (606, 431),
    (676, 431)
])
# Mapping from those points to a rectangle for a birdseye view.
DST = np.float32([
    (200, 720),
    (1080, 720),
    (200, 0),
    (1080, 0)
])