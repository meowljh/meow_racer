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


class Toy_Car(Car):
    def __init__(self, world, init_angle, init_x, init_y,
                 do_reverse:bool=False):
        super(Car, self).__init__() # world=world, init_angle=init_angle, init_x=init_x, init_y=init_y)
        self.init_x = init_x
        self.init_y = init_y
        self.world = world
        # breakpoint()
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(shape=polygonShape(
                    vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY1]
                ), density=1.),
                fixtureDef(shape=polygonShape(
                    vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY2]
                ), density=1.),
                fixtureDef(shape=polygonShape(
                    vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY3]
                ), density=1.),
                fixtureDef(shape=polygonShape(
                    vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY4]
                ), density=1.),
            ]
        )
        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        '''바퀴의 위치는 앞/뒤에 전혀 관련이 없었다. 예상했던 대로 dynamic body build를 할 때의 angle에 따라서 차량의 이동 방향이 바뀌었다.
        그런데 track를 만들 때의 alpha값이 radian이었는데 계속 180을 더해서 이상하게 계산이 됬었다.
        math.pi로 더해주니까 원하는대로 역방향으로 이동하였다.'''
        # if do_reverse:
        #     WHEELPOS = WHEELPOS[2:] + WHEELPOS[:2]
        for wx, wy in WHEELPOS:
            front_k = 1.0
            w = self.world.CreateDynamicBody(
                # 중심으로부터의 거리 #
                position=(init_x + wx * SIZE, init_y + wy * SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[(x*front_k*SIZE, y*front_k*SIZE) for x,y in WHEEL_POLY]
                    ),
                    density=0.1,
                    categoryBits=0x0020, maskBits=0x001, restitution=0.8
                )
            )
            
            w.wheel_rad = front_k * WHEEL_R * SIZE
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
                localAnchorA=(wx*SIZE, wy*SIZE), # the point in body A around which it will rotate #
                localAnchorB=(0, 0), # the point in body B around which it will rotate #
                enableMotor=True, # whether the joint motor will be active #
                enableLimit=True, # whether the joint limits will be active #
                maxMotorTorque=180 * 900 * SIZE * SIZE, # the maximum allowable torque the motor can use #
                motorSpeed=0, # the target speed of the joint motor #
                lowerAngle=-0.4, # angle for the lower limit #
                upperAngle=0.4 # angle for the upper limit #
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
    

    def gas(self, gas):
        """ vehicle control (gas) => 뒷바퀴만 움직이게 함. (Rear wheel drive)
        Args:
            gas (float): How much gas gets applied. Will be clipped between (0, 1)
                        think of it as the APS value
        """
        gas = np.clip(gas, 0., +1.)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1 # gas value는 너무 급속도로 증가하지 않도록 한다.#
            w.gas += diff
            self.control_logs[f"wheel_{w}"].append(diff)
    
    def brake(self, brake):
        """vehicle control (brake)
        Args:
            brake (float): Degree to which the brakes are applied - 즉, brake 페달을 밟는 각도를 의미. 0.9를 넘으면 바퀴가 회전하지 못함.
                        (0., 1.) 사이의 값을 가짐
        """
        # brake = np.clip(brake, 0., +1.)
        for w in self.wheels:
            w.brake = brake
            self.control_logs[f"wheel_{w}"].append(brake)
    
    def steer(self, steer):
        """Vehicle control (steer)
        Args:
            steer (float): target position 
                        (-1., 1.) 사이의 값을 가짐
                        
                        앞바퀴만 조향을 변경 
                        차량의 입력 조향각
        """ 
        # steer = np.clip(steer, -1., +1.)
        for i in range(2):
            self.wheels[i].steer = steer
            self.control_logs[f"wheel_{i}"].append(steer)
     
    def step(self, dt, new_friction=None):
        # breakpoint()
        from envs.gym_car_constants import FRICTION_LIMIT, ENGINE_POWER, WHEEL_MOMENT_OF_INERTIA
        # if use_jw:
        #     FRICTION_LIMIT = JW_FRICTION_LIMIT
        #     ENGINE_POWER = JW_ENGINE_POWER
        #     WHEEL_MOMENT_OF_INERTIA = JW_WHEEL_MOMENT_OF_INERTIA
        #     force_scale = JW_FORCE_SCALE
        # else:
        FRICTION_LIMIT = float(FRICTION_LIMIT) * new_friction # if new_friction is not None else FRICTION_LIMIT
        ENGINE_POWER = float(ENGINE_POWER) * new_friction # if new_friction is not None else ENGINE_POWER
        WHEEL_MOMENT_OF_INERTIA = float(WHEEL_MOMENT_OF_INERTIA) * new_friction  # if new_friction is not None else WHEEL_MOMENT_OF_INERTIA
        force_scale = 205000 * SIZE * SIZE * new_friction * new_friction # if new_friction is not None else 205000 * SIZE*SIZE ##scaling factor on the forces to prevent oscillation while training (especially when starting)##
        
        
        for w in self.wheels:
            #### STEP 1: Steer each wheel ####
            steering_dir = np.sign(w.steer - w.joint.angle)
            steering_val = abs(w.steer - w.joint.angle)
            ## the driving wheels of a car or vehicle can be simulated by changing the direction and size of the motor speed, usually setting the target speed to zero when car stops ##
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
            vf = hori_force[0] * v[0] + hori_force[1] * v[1] # speed vector forward #
            vs = vert_force[0] * v[0] + vert_force[1] * v[1] # speed vector sidewards #
            
            w.v = v
            w.vf = vf # 종방향 속도 #
            w.vs = vs # 횡방향 속도 #
            
            ##### STEP 4: Angular Velocity of the wheel #####
            # w.omega = 바퀴의 각속도 #
            ## 바퀴의 각속도에 대한 적분 식을 간단하게 풀기 위해서 dt값을 곱해줌.
            ## [0., 1.]의 범위를 갖는 w.gas값은 엔진이 생성할 수 있는 토크에 비례적인 값이다.
            ## ENGINE_POWER * w.gas / WHEEL_MOMENT_OF_INERTIA = 바퀴의 각속도 
            w.omega += (
                dt 
                * ENGINE_POWER 
                * w.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.)
            ) # omega값이 0일것을 대비해서 작은 값(5.)을 더해줌 #
            self.fuel_spent += dt * ENGINE_POWER * w.gas
            
            ##### STEP 5: Control the angular velocity of the wheel ######
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
            
            
            f_force = (-vf + vr) * force_scale
            p_force = (-vs) * force_scale
               
            force = np.sqrt(np.square(f_force) + np.square(p_force)) # absolute force that the wheel gets #
            
            #### STEP 6: Skid trace ####
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
                    (-WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE), (+WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                    (+WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE), (-WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE)
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