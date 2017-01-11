import getopt
import sys
from os.path import join, basename

import utils
import constants as c


def run(input_path):
    # Read input
    imgs = utils.read_input(input_path)

    # Calibrate camera
    camera_mat, dist_coeffs = utils.calibrate_camera()

    # Correct for distortion
    imgs_undistorted = utils.undistort_imgs(imgs, camera_mat, dist_coeffs)

    # Get masks
    masks = utils.get_masks(imgs_undistorted)

    # Transform perspective
    masks_birdseye = utils.birdseye(masks)  # TODO: Maybe not good enough.

    # # Find lines and curvature
    # l_r_c = utils.find_lines(masks_birdseye)
    #
    # # Draw lane
    # imgs_superimposed = utils.draw_lane(imgs, l_r_c)
    #
    # # Output image / video
    # save_path = join(c.SAVE_DIR, basename(input_path))  # Use same filename as input, but in save directory.
    # utils.save(imgs_superimposed, save_path)

    import numpy as np
    imgs_final = masks_birdseye
    return imgs_final

##
# TEST
##

from glob import glob
def test():
    paths = glob(join(c.TEST_DIR, 'test*.jpg'))
    for path in paths:
        img = run(path)

        save_path = join(c.SAVE_DIR, 'test_' + basename(path))
        utils.save(img * 255, save_path)


##
# Handle command line input
##

def print_usage():
    print 'Usage:'
    print '(-p / --path=) <path/to/image/or/video>'
    print '(-T / --test)  (Boolean flag. Whether to run the test function instead of normal run.)'

if __name__ == "__main__":
    path = None

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'p:T', ['path=', 'test'])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-p', '--path'):
            path = arg
        if opt in ('-T', '--test'):
            test()
            sys.exit(2)

    if path is None:
        print_usage()
        sys.exit(2)

    run(path)