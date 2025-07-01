import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from collections import defaultdict
import pygame
import Box2D
from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef


from envs.org_car import Car
from envs.gym_car_constants import *


class JW_Toy_Car(Car):
    def __init__(self, world, init_angle, init_x, init_y,
                 do_reverse:bool=False,
                 use_beta_dist:bool=False):
        super(Car, self).__init__() # world=world, init_angle=init_angle, init_x=init_x, init_y=init_y)
        self.use_beta_dist = use_beta_dist
        
        self.do_reverse = do_reverse
        self.init_x = init_x
        self.init_y = init_y
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(shape=polygonShape( 
                    vertices=[(x*JW_SIZE, y*JW_SIZE) for x,y in JW_HULL_POLY1]
                ), density=1.),
                ##########
                # fixtureDef(shape=polygonShape(
                #     vertices=[(x*JW_SIZE, y*JW_SIZE) for x, y in HULL_POLY1]
                # ), density=1.),
                # fixtureDef(shape=polygonShape(
                #     vertices=[(x*JW_SIZE, y*JW_SIZE) for x, y in HULL_POLY2]
                # ), density=1.),
                # fixtureDef(shape=polygonShape(
                #     vertices=[(x*JW_SIZE, y*JW_SIZE) for x, y in HULL_POLY3]
                # ), density=1.),
                # fixtureDef(shape=polygonShape(
                #     vertices=[(x*JW_SIZE, y*JW_SIZE) for x, y in HULL_POLY4]
                # ), density=1.),
            ]
        )
        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        from envs.gym_car_constants import WHEELPOS as wheel_pos
 
        # if self.do_reverse:
        #     WHEELPOS = wheel_pos[2:] + wheel_pos[:2]
        # else:
        #     WHEELPOS = wheel_pos
        for wx, wy in WHEELPOS:
            front_k = 1.0
            w = self.world.CreateDynamicBody(
                # 중심으로부터의 거리 #
                position=(init_x + wx * JW_SIZE, init_y + wy * JW_SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[(x*front_k*JW_SIZE, y*front_k*JW_SIZE) for x,y in JW_WHEEL_POLY]
                        # vertices=[(x*front_k*JW_SIZE, y*front_k*JW_SIZE) for x,y in WHEEL_POLY]
                        
                    ),
                    density=0.1, categoryBits=0x0020, maskBits=0x001, restitution=0.8
                )
            )
            
            w.wheel_rad = front_k * JW_WHEEL_R * JW_SIZE
            w.color = WHEEL_COLOR
            w.v  = 0.
            w.vs = 0.
            w.vf = 0.
            w.f_force = 0.
            w.p_force = 0.
            w.force = 0.
            
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0 # wheel angle #
            w.omega = 0.0 # 각속도 #
            w.skid_start = None
            ''' 'b2RevoluteJointDef': ['anchor', 'bodyA', 'bodyB', 'collideConnected', 'enableLimit', 
                                      'enableMotor', 'localAnchorA', 'localAnchorB', 'lowerAngle', 
                                      'maxMotorTorque', 'motorSpeed', 'referenceAngle', 'type', 
                                      'upperAngle', 'userData', ],'''
            rjd = revoluteJointDef(
                bodyA=self.hull, # car body chassi object #
                bodyB=w, # wheel #
                localAnchorA=(wx*JW_SIZE, wy*JW_SIZE), # the point in body A around which it will rotate #
                localAnchorB=(0, 0), # the point in body B around which it will rotate #
                enableMotor=True, # whether the joint motor will be active #
                enableLimit=True, # whether the joint limits will be active #
                # maxMotorTorque=100 * 900 * JW_SIZE * JW_SIZE, # the maximum allowable torque the motor can use #
                maxMotorTorque= JW_MAX_MOTOR_TORQUE,
                lowerAngle=JW_ANGLE_MIN, # angle for the lower limit #
                upperAngle=JW_ANGLE_MAX # angle for the upper limit #
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.limit_tiles = set()
            w.userData = w
            self.wheels.append(w)
        
        self.drawlist = self.wheels + [self.hull]
        self.particles = []
        
        self.control_logs = defaultdict(list)
        self.state_logs = defaultdict(list)

        self.prev_steer = 0
        self.prev_gas = 0
        self.prev_brake = 0
        

    def gas(self, gas):
        """ vehicle control (gas) => 뒷바퀴만 움직이게 함. (Rear wheel drive)
        Args:
            gas (float): How much gas gets applied. Will be clipped between (0, 1)
                        think of it as the APS value
        """
        gas = np.clip(gas, 0., +1.)
        # gas = 1. ### DEBUG ###
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1 # gas value는 너무 급속도로 증가하지 않도록 한다.#
            w.gas += diff
            self.control_logs[f"wheel_{w}"].append(diff)
            
        self.prev_gas = gas
    
    def brake(self, brake):
        """vehicle control (brake)
        Args:
            brake (float): Degree to which the brakes are applied - 즉, brake 페달을 밟는 각도를 의미. 0.9를 넘으면 바퀴가 회전하지 못함.
                        (0., 1.) 사이의 값을 가짐
        """ 
        # brake = 0. ### DEBUG ###
        for w in self.wheels:
            w.brake = brake
            self.control_logs[f"wheel_{w}"].append(brake)
        
        self.prev_brake = brake
    
    def steer(self, steer):
        """Vehicle control (steer)
        Args:
            steer (float): target position 
                        (-1., 1.) 사이의 값을 가짐
                        
                        앞바퀴만 조향을 변경 (원래 dynamics bicycle model을 보면 이런 형태로 구현이 됨. 후륜 구동으로, 뒷바퀴가 "주행"을 하게 하고, 앞바퀴에 의해서 "조향"이 결정된다.)
                        차량의 입력 조향각
        """  
        # steer = 0. ### DEBUG ###
        # breakpoint()
        # if self.use_beta_dist:
        #     steer = (steer + 0.5) * 2
            
        for i in range(2):
            self.wheels[i].steer = steer
            self.control_logs[f"wheel_{i}"].append(steer)

        # self.prev_steer = steer
        self.prev_steer = -steer # action 넣어줄때 -를 곱해주기 때문
        
    def step(self, dt, new_friction=None, use_jw:bool=False):
        # breakpoint()
        from envs.gym_car_constants import FRICTION_LIMIT, ENGINE_POWER, WHEEL_MOMENT_OF_INERTIA
        FRICTION_LIMIT = JW_FRICTION_LIMIT
        ENGINE_POWER = JW_ENGINE_POWER
        WHEEL_MOMENT_OF_INERTIA = JW_WHEEL_MOMENT_OF_INERTIA
        force_scale = JW_FORCE_SCALE
        
        # for w in self.wheels:
        for i, w in enumerate(self.wheels):
            #### STEP 1: Steer each wheel ####
            steering_dir = np.sign(w.steer - w.joint.angle)
            steering_val = abs(w.steer - w.joint.angle) ## -1~+1 -> joint.angle (-0.4 ~ +0.4) -> (1.6)
            # if i == 0:
            #     breakpoint()
            ## the driving wheels of a car or vehicle can be simulated by changing the direction and size of the motor speed, usually setting the target speed to zero when car stops ##
            ### 여기서는 50 degrees/s로 둔다. 그럼 1rad = 2pi이기 때문에 분당 회전수인 RPM은 50rad/s / 2pi * 60을 해야 구할 수 있게 된다.
            '''motorSpeed: 차량의 조향의 회전 속도 (조향 모터가 얼마나 빠르게 바퀴의 방향을 돌릴지 결정하는 값)
            -> 따라서 바퀴의 조향(joint)가 적용 대상이다. -> 이를 바탕으로 w.joint.angle이 회전함.
            
            steering_dir: 조향 방향(목표 조향각인 w.steer로 현재 조향각인 w.joint.angle을 바꾸기 위해 조향해야 하는 방향)
            steering_val: 조향 크기
            현재 조향각을 목표 조향각인 w.steer로 이동시키도록 모터 속도를 설정한다.
            max motor speed의 제한이 3rad/sec이기 때문에 약 171deg/sec이 되는 것이다.'''
            w.joint.motorSpeed = steering_dir * min(50. * steering_val, 3.)
            
            #### STEP 2: Consider the friction of the grass / track ####
            # 근데 지금은 그냥 잔디랑 실제 도로의 마찰력을 굳이 반영할 필요는 없어보임.. ##
            grass = True
            friction_limit = FRICTION_LIMIT * 0.6 # use the grass friction if there is no tile #
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, FRICTION_LIMIT * tile.road_friction
                )
                grass = False
            # friction_limit = FRICTION_LIMIT
            # grass=False
                  
            #### STEP 3: Force on wheel ####
            hori_force = w.GetWorldVector((0,1)) # 종방향 힘 #
            vert_force = w.GetWorldVector((1,0)) # 횡방향 힘 #
            v = w.linearVelocity # 차량의 이동방향으로의 속도 v_car #
            vf = hori_force[0] * v[0] + hori_force[1] * v[1] # speed vector forward (바퀴가 진행하는 방향의 속도) #
            vs = vert_force[0] * v[0] + vert_force[1] * v[1] # speed vector sidewards (바퀴의 측면 속도) #
            
            w.v = v
            w.vf = vf # 종방향 속도 #
            w.vs = vs # 횡방향 속도 #
            
            ##### STEP 4: Angular Velocity of the wheel #####
            # w.omega = 바퀴의 각속도 #
            ## 바퀴의 각속도에 대한 적분 식을 간단하게 풀기 위해서 dt값을 곱해줌.
            ## [0., 1.]의 범위를 갖는 w.gas값은 엔진이 생성할 수 있는 토크에 비례적인 값이다.
            ## ENGINE_POWER * w.gas / WHEEL_MOMENT_OF_INERTIA = 바퀴의 각속도 
            # ENGINE_POWER * w.gas = 현재 사용자가 낼 수 있는 엔진의 출력 (=힘)
            '''차량이 가속할 때의 엔진의 출력을 바퀴에 전달하여 회전 속도인 w.omega를 증가시킴'''
            w.omega += (
                dt 
                * ENGINE_POWER 
                * w.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.)
            ) # omega값이 0일것을 대비해서 작은 값(5.)을 더해줌 #
            self.fuel_spent += dt * ENGINE_POWER * w.gas
            
            ##### STEP 5: Control the angular velocity of the wheel ######
            '''강한 brake 입력(>0.9)이면 즉시 정지, 아니면 브레이크의 입력에 따른 감속량 계산'''
            if w.brake >= 0.9:
                self.omega = 0 # brake강도가 0.9를 넘으면 각속도 없음 #
            elif w.brake > 0:
                BRAKE_FORCE = 15 # radians per second => 각속도 omega와 단위가 동일 #
                dir = -np.sign(w.omega) # 바퀴의 현재 각속도의 반대 방향 #
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega): # brake를 더 세게 밟는 경우에 저속 #
                    val = abs(w.omega) # 이 경우에 omega += dir*val을 하면 각속도가 brake에 의해서 0이 된다고 보면 됨 #
                w.omega += dir * val 
            w.phase += w.omega * dt # 움직인 radian의 크기 #
            
            ###### STEP 6: Calculate the speed of the wheel ######
            vr = w.omega * w.wheel_rad # 바퀴 선 속도 (각속도 * 휠 반지름) -> 바퀴의 표면(접지면)이 이동하는 선형 거리, linear distance 계산 가능 #
            
            '''
            v_r: 바퀴의 회전 속도에 따른 선형 속도
            f_force: 노면 속도와 바퀴의 회전 속도의 차이에서 발생하는 힘
            p_force: 측면으로 미끄러지는 속도에 의한 힘
          
            
            if vr > vf: f_force > 0 -> 차량이 앞으로 전진
            else: f_force < 0 -> 감속 및 슬립
            vs: 측면 속도(=슬립 속도) -> 따라서 p_force는 슬립의 반대방향으로 작용하는 마찰력
            '''
            f_force = (-vf + vr) * force_scale #### 지면과의 마찰력으로 작용 ####
            p_force = (-vs) * force_scale
               
            force = np.sqrt(np.square(f_force) + np.square(p_force)) # absolute force that the wheel gets #
            
            #### STEP 6: Skid trace ####
            ## this part is only used for creating the skidding particles, and not related to the vehicle forces or dynamics ##
            '''물리적으로 허용된 마찰력을 초과하면 조정하여 과도한 힘 적용 방지'''
            if abs(force) > 2. * friction_limit:
                if (
                    w.skid_particle and w.skid_particle.grass == grass and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                    
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None
            
            if abs(force) > friction_limit:  # if total force is more than the friction limit, then the forces are scaled to match exactly the friction limit #
                f_force /= force
                p_force /= force
                force = friction_limit
                f_force *= force
                p_force *= force
            
            ##### STEP 7: Consider the friction force applied to the wheel ######
            ## 마찰력에 의한 바퀴의 각속도 감소를 고려
            # 마챨력에 의한 각가속도 = 마찰력 (=마찰 토크) * 반지름 / 관성 모먼트
            '''바퀴의 회전 속도인 w.omega를 마찰력에 의해 감소'''
            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA
            
            w.p_force = p_force # propulsion force (차량의 엔진, 모터가 바퀴에 전달하는 추진력) -> 바퀴의 토크와 반지름을 기반으로 계산이 됨. 차량을 앞으로 움직이는 힘 #
            w.f_force = f_force # friction force (바퀴와 노면 사이에서 발생하는 마찰력) -> 차량이 가속하거나 감속하는데 필요한 집지력 #
            w.force = force
            
            w.ApplyForceToCenter(
                (
                    p_force * vert_force[0] + f_force * hori_force[0],
                    p_force * vert_force[1] + f_force * hori_force[1]
                ), True
            )
            
    def draw(self, surface, zoom, translation, angle, draw_particles:bool=True):
        
        if draw_particles:
            for p in self.particles:
                poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]            
                poly = [
                    (coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in poly
                ]
                pygame.draw.lines(
                    surface, color=p.color, points=poly, width=2, closed=False
                )
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [
                    (coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in path
                ]
                
                color = [int(c * 255) for c in obj.color]
                pygame.draw.polygon(surface, color=color, points=path)

                if "phase" not in obj.__dict__: # 바퀴가 아닌 chassi body인 경우 #
                    continue
                a1 = obj.phase
                a2 = obj.phase + 1.2 # radians #
                s1, c1, s2, c2 = math.sin(a1), math.cos(a1), math.sin(a2), math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                white_poly = [
                    (-JW_WHEEL_W * JW_SIZE, +JW_WHEEL_R * c1 * JW_SIZE), (+JW_WHEEL_W * JW_SIZE, +JW_WHEEL_R * c1 * JW_SIZE),
                    (+JW_WHEEL_W * JW_SIZE, +JW_WHEEL_R * c2 * JW_SIZE), (-JW_WHEEL_W * JW_SIZE, +JW_WHEEL_R * c2 * JW_SIZE)
                ]
                white_poly = [trans*v for v in white_poly]

                white_poly = [(coords[0], coords[1]) for coords in white_poly]
                
                white_poly = [
                    pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly
                ]
                white_poly = [
                    (coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in white_poly
                ]
                pygame.draw.polygon(surface, color=WHEEL_WHITE, points=white_poly)
 
    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass
        
        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p
    
    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []