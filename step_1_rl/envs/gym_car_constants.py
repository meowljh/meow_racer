import math
import numpy as np
########################################################################
######################## NAM ENVIRONMENT ###############################
########################################################################
NAM_SCALE = 6.0
NAM_WINDOW_W = 800 #  1000
NAM_WINDOW_H = 800 # 1000
NAM_ZOOM = 2.7
NAM_FPS = 50
NAM_TRACK_DETAIL_STEP = 14 / NAM_SCALE
NAM_TRACK_TURN_RATE = 0.31
NAM_BORDER_MIN_COUNT = 7
NAM_TRACK_WIDTH = 40 / NAM_SCALE

REAL_NAM_TRACK_ENTIRE_WIDTH = 42 / NAM_SCALE
REAL_NAM_TRACK_HALF_WIDTH = REAL_NAM_TRACK_ENTIRE_WIDTH / 2

NAM_BORDER = 8 / NAM_SCALE
NAM_PLAYFIELD = 40000 / NAM_SCALE
NAM_GRASS_DIM = NAM_PLAYFIELD / 20.0
NAM_MAX_SHAPE_DIM = (
    max(NAM_GRASS_DIM, NAM_TRACK_WIDTH, NAM_TRACK_DETAIL_STEP) * math.sqrt(2) * NAM_ZOOM * NAM_SCALE
)
########################################################################
######################## MAP ENVIRONMENT ###############################
########################################################################
STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 400 # 600
VIDEO_H =  600 # 400
WINDOW_W = 800 # 1000 # 800 # 900 # 1000 # 1000
WINDOW_H = 800 # 1000 # 800 # 900 # 1000 # 800

CHECKPOINTS = 12 # 18 ## checkpoint도 줄여서 더 쉬운 트랙으로 만들자. 18개는 너무 많았고, PID 제어를 할 때에도 속도가 50을 넘어도 스핀이 발생했다. ##
SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 14 / SCALE # 21 / SCALE  # 5 / SCALE # 21 / SCALE
TRACK_DETAIL_STEP_CHECK = 21 / SCALE
TRACK_TURN_RATE = 0.31 ## pi값보단 작게 돌아야 한다는 의미임. 근데 이렇게 되면 트랙이 너무 쉬워지는 느낌임.
TRACK_WIDTH = 40 / SCALE ## 트랙 반절 너비 , 즉 트랙을 구성하는 tile 하나의 너비의 반절##

REAL_TRACK_ENTIRE_WIDTH = 42 / SCALE
REAL_TRACK_HALF_WIDTH = REAL_TRACK_ENTIRE_WIDTH / 2

OUT_TRACK_LIMIT = 80 / SCALE # 90 / SCALE ## 트랙 밖으로 벗어나는 경우에 허용해 주는 경계선의 범위 ##

BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)
########################################################################
######################## CAR ENVIRONMENT ###############################
########################################################################

SIZE = 0.02 
ENGINE_POWER = 100000000 * SIZE * SIZE # 550HP -> 너무 큼 ##
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE
FRICTION_LIMIT = (
    1000000 * SIZE * SIZE
)  # friction ~= mass ~= size^2 (calculated implicitly using density)



WHEEL_R = 27 # wheel 바깥 반지름 
WHEEL_W = 14 # wheel 안쪽 반지름
# 앞축 왼쪽 - 앞축 오른쪽 - 뒷축 왼쪽 - 뒷축 오른쪽
WHEEL_POLY = [(-WHEEL_W, +WHEEL_R), (+WHEEL_W, +WHEEL_R), (+WHEEL_W, -WHEEL_R), (-WHEEL_W, -WHEEL_R)]
WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]


###################################################################################
######################### COMMON CAR ENVIRONMENT ##################################
###################################################################################

HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, 20)]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20),
]
HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]


###################################################################################
######################### JW CAR ENVIRONMENT ######################################
## debugging 과정을 통해서 gas를 1, brake를 0으로 계속 넣어 줄 때 속도가 30m/s이상으로 증가하는 것 확인.
## 때문에 friction limit이나 engine power을 아래와 같이 넣어주는게 가능
## max torque, max RPM은 정확히 알 수 있던 값이 아니어서 기존 car racing에서 사용한 차량의 파라미터 값을 그대로 사용하였다
# 어차피 friction limit에 걸리는 제한에 의해서 차량에 slip이 발생하기 전까지의 견딜 수 있는 최대 힘을 제한을 주었기 때문에 문제 없을거라 판단 (질량 * 마찰 계수 / 중력가속도)
###################################################################################
#### FRICTION_LIMIT = 마찰계수 * 수직하중 = mu * mass * g = 1 * 1840kg * 9.81m/s^2
# 아무래도 기존 environment의 car racing 환경에서 사용된 차량은 F1 차량같은거였는데, 이것보다 크기 때문에 견딜 수 있는 최대 마찰력은 당연히 커져야 한다고 생각함.
JW_SIZE = 0.02 # 0.01 # 0.02
JW_MASS = 1840 # 단위: kg #
JW_GRAVITY = 9.81 # 단위: m/s^2 #
JW_FRICTION_COEF =  1. #  마찰계수
JW_FRICTION_LIMIT =  4 * JW_MASS / JW_GRAVITY * JW_FRICTION_COEF # 약 187N #
JW_WHEEL_MOMENT_OF_INERTIA = 1.658 # (원래 값이랑 비슷) 2.254 # 3.6 # 0.36 #  1.8 # 1.45 # 단위: kg/m^2 # => 0.5 * M * (W_f^2 + W_r^2)
JW_ENGINE_POWER = 1000_000_00 * JW_SIZE * JW_SIZE  # 40000 -> 엔진 출력은 최대로 잡지만, friction limit에 제한을 두어서 바퀴에 주는 힘에 limit이 걸리도록 한다.
##최대로 걸릴 수 있는 모터 토크를 줄여야 JW 설정으로 할 때 최대 속도를 제한 할 수 있을 것 같음.
JW_MAX_MOTOR_TORQUE = 48 #  64.8 #  36 # 18 ## N/m ## -> MAX_MOTOR_TORQUE for original car was 36
JW_FORCE_SCALE = 41 # 20 # 205000 * JW_SIZE * JW_SIZE #  41 # 1.2 ## just a random number -> no influence on the friction limit ##

###바퀴의 크기를 WHEEL_R > WHEEL_W이 되도록 했어야 했는데.. 실수함 ㅎ ####
JW_WHEEL_R =  24 # 36 # 24 # 18 # 36 # 35 # 35 # 54 # 27 # wheel 바깥 반지름 
JW_WHEEL_W =   18 # 10 # 24 # 48 # 24 # 28 # 28 # 14 # wheel 안쪽 반지름
JW_WHEEL_MASS = 12.86
#### minimum & maximum valid slip angle 사용 하였을 때 계속 spin을 하였기 때문에 이 값보다 작은 값으로 바꿔줄 필요가 있었음.
JW_ANGLE_MIN = -0.4 # -1.5 ## 약 85 -> 55 degrees
JW_ANGLE_MAX = +0.4 # +1.5 ## 약 -85 -> -55 degrees
# 앞축 왼쪽 - 앞축 오른쪽 - 뒷축 왼쪽 - 뒷축 오른쪽
# JW_HULL_POLY1 =  [(-25, +50), (+25, +50), (+25, -50), (-25, -50)]
# JW_WHEEL_POS = [(-22.5, +30), (+22.5, +30), (-22.5, -30), (+22.5, -30)]
###차량의 polygon의 좌표가 제대로 설정되지 않았기 때문에 문제가 발생하는 것일수도 있음###
JW_HULL_POLY1 = [(-50, +100), (+50, +100), (+50, -100), (-50, -100)]
JW_WHEEL_POS = [(-45, +80), (+45, +80), (+45, -80), (-45, -80)] #전륜 2개, 후륜 2개#
JW_CAR_WIDTH = 100 * JW_SIZE
JW_WHEEL_POLY = [(-JW_WHEEL_W, +JW_WHEEL_R), (+JW_WHEEL_W, +JW_WHEEL_R), (+JW_WHEEL_W, -JW_WHEEL_R), (-JW_WHEEL_W, -JW_WHEEL_R)]
###################################################################################
WHEEL_COLOR = (0, 0, 0)
WHEEL_WHITE = (77, 77, 77)
MUD_COLOR = (102, 102, 0)
ROAD_COLOR =  np.array([102, 102, 102])  # np.random.uniform(0, 210, size=3) # np.random.uniform(120, 210, size=3)
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (255, 0, 0)

