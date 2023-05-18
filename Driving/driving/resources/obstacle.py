import pybullet as p
import os
import numpy as np


class Obstacle:
    """
    Class defining the Obstacle in the simulation environment.
    """

    def __init__(self, client, base):
        """
        Initializes the Obstacle object.

        Parameters:
        - client: the id of the physics client,
        - base: the position of the obstacle.
        """

        # Choose between 'obstacle1.urdf' and 'obstacle2.urdf' randomly
        if np.random.rand() < 0.5:
            f_name = os.path.join(os.path.dirname(__file__), 'obstacle1.urdf')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'obstacle2.urdf')

        # Load the URDF model of the obstacle
        self.obstacle = p.loadURDF(fileName=f_name,
                                   basePosition=[base[0], base[1], 0],
                                   physicsClientId=client)

        # Save the position of the obstacle
        self.pos = base

    def get_id(self):
        """
        Returns the id of the obstacle.

        Returns:
        - the id of the obstacle.
        """

        return self.obstacle

    def get_pos(self):
        """
        Returns the position of the obstacle.

        Returns:
        - the position of the obstacle.
        """

        return self.pos
