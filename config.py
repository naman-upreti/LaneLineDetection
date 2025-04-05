# Configuration parameters for lane detection
import numpy as np

# Video processing
VIDEO_WIDTH = 600
VIDEO_HEIGHT = 500
FRAME_RATE = 10

# Lane detection parameters
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 0

# Hough transform parameters
HOUGH_RHO = 4
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LENGTH = 100
HOUGH_MAX_LINE_GAP = 180

# Line filtering
MIN_SLOPE = 0.4
MAX_SLOPE = 2.0
LINE_SMOOTHING_FACTOR = 0.2

# Region of interest
def get_roi_vertices(image_shape):
    height, width = image_shape[:2]
    return np.array([
        [(width / 9, height),
         (width / 2 - width / 8, height / 2 + height / 10),
         (width / 2 + width / 8, height / 2 + height / 10),
         (width - width / 9, height)]
    ], dtype=np.int32)

# Color thresholds
YELLOW_HSV_LOW = (20, 100, 100)
YELLOW_HSV_HIGH = (30, 255, 255)
WHITE_THRESHOLD_LOW = 200
WHITE_THRESHOLD_HIGH = 255

# Image blending
BLENDING_ALPHA = 0.8
BLENDING_BETA = 1.0
BLENDING_GAMMA = 0.0