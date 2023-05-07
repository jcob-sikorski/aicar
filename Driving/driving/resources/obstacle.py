import pybullet as p
import os
import numpy as np


# class Goal:
#     def __init__(self, client, base):
#         f_name = os.path.join(os.path.dirname(__file__), 'goal.urdf')
#         p.loadURDF(fileName=f_name,
#                    basePosition=[base[0], base[1], 0],
#                    physicsClientId=client)
        

class Obstacle:
    def __init__(self, client, base):
        if np.random.rand() < 0.5:
            f_name = os.path.join(os.path.dirname(__file__), 'obstacle1.urdf')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'obstacle2.urdf')
            
        p.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0],
                   physicsClientId=client)