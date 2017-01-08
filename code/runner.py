import getopt
import sys

import utils
import constants as c

def run(path):
    ##
    # TODO: Read input
    ##
    imgs = utils.read_input(path)

    ##
    # TODO: Calibrate camera
    ##
    camera_mat, dist_coeffs = utils.calibrate_camera()

    ##
    # TODO: Correct for distortion
    ##
    imgs_undistorted =

    ##
    # TODO: Mask images
    ##

    ##
    # TODO: Perspective transform
    ##

    ##
    # TODO: Find lines and curvature
    ##

    ##
    # TODO: Output image / video
    ##

def print_usage():
    print 'Usage:'
    print '(-p / --path=) <path/to/image/or/video>'

if __name__ == "__main__":
    path = None

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'p:', ['path='])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-p', '--path'):
            path = arg
        else:
            print_usage()
            sys.exit(2)

    run(path)