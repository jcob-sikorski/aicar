import pybullet as p
import os
import math
import numpy as np

class Car:
    """
    Class defining the behavior of the car in the simulation environment.
    """

    def __init__(self, client, base, orientation=0):
        """
        Initializes the car object.

        Parameters:
        - client: the id of the physics client,
        - base: the starting position of the car,
        - orientation: the starting orientation of the car.
        """
        self.client = client
        self.orientation = orientation

        # Convert the orientation angle to a quaternion
        self.orientation_quat = p.getQuaternionFromEuler([0, 0, self.orientation])

        # Load the URDF model of the car
        f_name = os.path.join(os.path.dirname(__file__), 'car.urdf')
        self.car = p.loadURDF(fileName=f_name,
                              basePosition=[base[0], base[1], 0.1],
                              physicsClientId=client,
                              baseOrientation=self.orientation_quat)

        # Define the joint indices related to the car's steering and driving
        self.steering_joints = [3, 5]
        self.drive_joints = [4, 6, 7, 8]

        # Set the initial speed of the car's joints
        self.joint_speed = 0

        # Define the drag constants to account for resistance due to friction
        self.c_rolling = 0.2
        self.c_drag = 0.01

        # Define the throttle constant to control the speed of the car
        self.c_throttle = 20

    def get_ids(self):
        """
        Returns the ids of the car and the physics client.
        """
        return self.car, self.client

    def apply_action(self, action):
        """
        Applies the given action to the car.

        The action consists of a throttle and steering angle, which are
        used to control the car's velocity and direction.

        Parameters:
        - action: a two-element list or array, where the first element
        represents the throttle and the second one the steering angle.
        """
        # Extract throttle and steering angle from action
        throttle, steering_angle = action[0], action[1]

        # Clip throttle and steering angle to their respective limits
        throttle = min(max(throttle, 1), 1.5)
        steering_angle = max(min(steering_angle, 135), -135)

        # Set the steering joint positions
        p.setJointMotorControlArray(self.car, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2,
                                    physicsClientId=self.client)

        # Calculate the friction due to drag and mechanical resistance
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        # Calculate the acceleration taking into account throttle and friction
        acceleration = self.c_throttle * throttle + friction

        # Update the joint speed
        self.joint_speed = self.joint_speed + 1/30 * acceleration
        if self.joint_speed < 0:
            self.joint_speed = 0

        # Set the velocity of the wheel joints directly
        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[1.2] * 4,
            physicsClientId=self.client)

    def get_observation(self, goal):
        """
        Returns an observation from the environment.

        The observation includes the car's position, orientation, velocity,
        and a camera image from the car's perspective.

        Parameters:
        - goal: the coordinates of the goal location.
        """
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)

        # Calculate the orientation of the car as a 2D vector
        ori = (math.cos(ang[2]), math.sin(ang[2]))

        # Get the velocity of the car
        vel = p.getBaseVelocity(self.car, self.client)[0][0:2]

        # Calculate the Euclidean distance to the goal
        dist_to_goal = math.sqrt(((pos[0] - goal[0]) ** 2 +
                                  (pos[1] - goal[1]) ** 2))

        # Define camera parameters
        width = 320  # Width of the camera image
        height = 240  # Height of the camera image
        fov = 60  # Field of view of the camera
        aspect_ratio = width / height
        near_plane = 0.1  # Near clipping plane
        far_plane = 100  # Far clipping plane

        # Define camera position and orientation
        cam_target = (pos[0] + ori[0], pos[1] + ori[1], pos[2])
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=cam_target,
            cameraUpVector=(0, 0, 1)
        )

        # Define the projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect_ratio,
            nearVal=near_plane,
            farVal=far_plane
        )

        # Get the camera image
        img_arr = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Extract the RGB image from the returned data
        rgb_image = np.reshape(
            np.array(img_arr[2], dtype=np.uint8),
            (height, width, 4)
        )[:, :, :3]
        
        return rgb_image
