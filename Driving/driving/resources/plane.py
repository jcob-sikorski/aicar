import pybullet as p
import pybullet_data


class Plane:
    """
    Class defining the Plane in the simulation environment.
    """

    def __init__(self, client):
        """
        Initializes the Plane object.

        Parameters:
        - client: the id of the physics client.
        """

        # Set the search path to include PyBullet data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the URDF model of the plane
        p.loadURDF('plane.urdf',
                   basePosition=[0, 0, 0],
                   physicsClientId=client)
