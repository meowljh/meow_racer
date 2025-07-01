import Box2D
from Box2D import *
from Box2D.b2 import contactListener, fixtureDef, polygonShape

class Detector(contactListener):
    '''    
    BeginContact = _swig_new_instance_method(_Box2D.b2ContactListener_BeginContact)
    EndContact = _swig_new_instance_method(_Box2D.b2ContactListener_EndContact)
    PreSolve = _swig_new_instance_method(_Box2D.b2ContactListener_PreSolve)
    PostSolve = _swig_new_instance_method(_Box2D.b2ContactListener_PostSolve)'''
    def __init__(self):
        super(Detector, self).__init__()
    
    
    def BeginContact(self, contact):
        self._contact(contact, True)
    
    def EndContact(self, contact):
        self._contact(contact, False)
    
    def _contact(self, contact, begin:bool):
        
class Tester(object):
    def __init__(self):
        super(Tester, self).__init__()
        self.detector = Detector()
        self.world = Box2D.b2World(gravity=(0, 0),
                                   contactListener=self.detector)

        body1 = self.world.CreateDynamicBody(position=(0, 0))
        body1.CreateCircleFixture()
        body2 = self.world.CreateDynamicBody(position=(0, 0))
        body2.CreateCircleFixture()

    def step(self):
        self.world.Step(1 / 30, velocityIterations=1, positionIterations=1)
        
    
if __name__ == "__main__":
    tester = Tester()
    for i in range(10):
        tester.step()