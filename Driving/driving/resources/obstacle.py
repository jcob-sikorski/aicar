import pybullet as p
import os
import numpy as np
        

class Obstacle:
    def __init__(self, client, base):
        if np.random.rand() < 0.5:
            f_name = os.path.join(os.path.dirname(__file__), 'obstacle1.urdf')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'obstacle2.urdf')
            
        self.obstacle = p.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0],
                   physicsClientId=client)
        
        self.pos = base

    def get_id(self):
        return self.obstacle
    
    def get_pos(self):
        return self.pos