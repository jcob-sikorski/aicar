import pybullet as p
import pybullet_data
import os


class Plane:
    def __init__(self, client):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF('plane.urdf',
                   basePosition=[0, 0, 0],
                   physicsClientId=client)
