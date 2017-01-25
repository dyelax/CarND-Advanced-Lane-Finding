import skvideo.io
import numpy as np

outputdata = np.random.random(size=(50, 480, 680, 3)) * 255
outputdata = outputdata.astype(np.uint8)

skvideo.io.vwrite("../output_images/outputvideo.mp4", outputdata)