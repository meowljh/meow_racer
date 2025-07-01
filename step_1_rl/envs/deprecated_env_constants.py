import numpy as np
import pickle
import math

meta_car_dict = pickle.load(open('statics/car_dict.pkl', 'rb'))
SCREEN_H = 600 # 1600 # 800 # float(meta_car_dict['screen_h'])
SCREEN_W = 400 # 1000 # 1000  #  float(meta_car_dict['screen_w'])

SCALE = 6.
PLAYFIELD = 2000 / SCALE #  max(SCREEN_H, SCREEN_W) * 2 / SCALE
GRASS_DIM = PLAYFIELD / 20.

NAM_SCALE = 6.
NAM_PLAYFIELD = 4800 / NAM_SCALE
NAM_GRASS_DIM = NAM_PLAYFIELD / 20.


FPS = 50 # frames per second #
ZOOM = 1.5 # 2.7 # camera zoom #

##### for random track generation #####
CHECKPOINTS = 12 # 12

TRACK_RAD = 900. / SCALE
TRACK_WIDTH = 40. / SCALE
TRACK_DETAIL_STEP = 21. / SCALE
BORDER = 8. / SCALE
BORDER_MIN_COUNT = 4
TRACK_TURN_RATE = 0.31

CAUTION_OFFSET = 16. / SCALE
 
##### for Nam-C track rendering #####

NAM_TRACK_WIDTH = 35 / 2 / NAM_SCALE #  3.5 # 3.5 # 7.
NAM_TRACK_TURN_RATE = 0.02
NAM_BORDER_MIN_COUNT = 20
NAM_BORDER = 10 / 2 / NAM_SCALE # 1 # s 1.

NAM_CAUTION_OFFSET = 20 / NAM_SCALE
NAM_CENTER = (0, 0) # (-50, 0)
 
##### for road coloring #####
ROAD_COLOR = np.array([102, 102, 102])
GRASS_COLOR = np.array([102, 230, 102])
RED_COLOR = np.array([255, 0, 0])
WHITE_COLOR = np.array([255, 255, 255])
BG_COLOR = np.array([102, 204, 102])
BLACK_COLOR = np.array([0, 0, 0])

NAM_MAX_SHAPE_DIM = (
    max(NAM_GRASS_DIM, NAM_TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * NAM_SCALE
)
MAX_SHAPE_DIM = (
     max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)