from os import makedirs
from os.path import join, exists

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
CALIBRATION_DATA = join(CALIBRATION_DIR, 'calibration_data.p')
TEST_DIR = join(DATA_DIR, 'test_images/')

SAVE_DIR = '../output_images/'

