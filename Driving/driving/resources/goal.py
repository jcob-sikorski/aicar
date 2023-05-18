import pybullet as p
import os

class Goal:
    """
    Class defining the Goal in the simulation environment.
    """

    def __init__(self, client, base):
        """
        Initializes the Goal object.

        Parameters:
        - client: the id of the physics client,
        - base: the position of the goal.
        """
        
        # Load the URDF model of the goal
        f_name = os.path.join(os.path.dirname(__file__), 'goal.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0],
                   physicsClientId=client)
