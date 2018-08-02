IMAGE_NAME = './b.jpg'
TARGET_NAME = 'lines.png' # Pre-processed LINES image
CLOSE_KERNEL_SIZE = (5, 5)
DILATE_KERNEL_SIZE = (5, 5)
'''HARRIS PARAMETERS'''
HARRIS_BLOCK_SIZE = 4
HARRIS_K_SIZE = 7
HARRIS_K =0.04
''''''
REDUCTION_DISTANCE_1 = 30 # Pixels
GRID_SIZE = 20 # Pixels
REDUCTION_DISTANCE_2 = 30 # Pixels
''''''
PIXEL_SHIFT = 0 # Shifts every corner N pixels up. Default is -1. Change this to any value greater than 0 to shift.
ALPHA = 0.2
BETA = 0.8
RANK_THRESHOLD = 0.35 # Sets threshold for Onur's Algorithm (alpha*similartiy + beta * 1/distance) > 0.35
''''''
SCALE = 10 # Scale the distance between vertices to be N times larger
REMOVE_COLLISION = True # Set this to True to remove colliding walls. Still in development. May produce bad output.