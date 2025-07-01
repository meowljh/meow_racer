import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) #008_
sys.path.append(ROOT)

from car_constants import *

import pygame

from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef

class Car:
    def __init__(self, world, vehicle_model, init_angle, init_x, init_y):
        super().__init__()
        self.world = world
        self.vehicle_model = vehicle_model
        
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = init_angle,
            fixtures = [
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x*CAR_SIZE, y*CAR_SIZE) for x,y in HULL_POLY]
                    ), density=1.0
                ), ## 우선은 간단하게 직사각형 모양으로 차량 대체. 바퀴만 제대로 구현하면 될듯 ##
                # fixtureDef(
                #     shape=polygonShape(
                #         vertices=[(x*CAR_SIZE, y*CAR_SIZE) for x,y in HULL_POLY1]
                #     ), density=1.0
                # ),
                # fixtureDef(
                #     shape=polygonShape(
                #         vertices=[(x*CAR_SIZE, y*CAR_SIZE) for x,y in HULL_POLY2]
                #     ), density=1.0
                # ),
                # fixtureDef(
                #     shape=polygonShape(
                #         vertices=[(x*CAR_SIZE, y*CAR_SIZE) for x,y in HULL_POLY3]
                #     ), density=1.0
                # )
            ]
        )
        self.hull.color = np.array(CAR_COLOR)
        self.wheels = []
        self.fuel_spent = 0.0
        for (wx, wy) in WHEEL_POS:
            
        self.drawlist = [self.hull]
        self.particles = []
         
    
    def apply_dynamics(self, action):
        delta = action[0] # 조향각 # (-0.5 ~ 0.5 사이의 값을 갖도록 -> 제한을 뒀다고 보면 됨)
        torque_ratio = action[1] # 토크값 # (-1 ~ 1 사이의 값을 갖도록)
        torque_val = abs(torque_ratio) * self.vehicle_model.torque_min if torque_ratio < 0 else abs(torque_ratio) * self.vehicle_model.torque_max
        
    
    def steer(self, s):
        """control steering wheel
        Args:
            s (-0.5 .. 0.5)
        Only controls the steering for the front wheel
        """
        self.wheels[0].steer = s
        self.wheels[1].steer = s
    
    def brake(self, b):
        """control brake
        Args:
            b (0..1)
        """
        for w in self.wheels:
            w.brake = b
            

    def draw(self, surface, zoom, translation, angle, draw_particles:bool=False):
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]  
                # breakpoint()
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1]
                    ) for coords in path
                ]  
                color = [int(c * 255) for c in obj.color]
                pygame.draw.polygon(surface, color=color, points=path)
                
    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []