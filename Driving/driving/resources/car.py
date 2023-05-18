import pybullet as p
import os
import math
import numpy as np


class Car:
    def __init__(self, client, base, orientation=0):
        self.client = client
        self.orientation = orientation
        # Convert the orientation angle to a quaternion
        self.orientation_quat = p.getQuaternionFromEuler([0, 0, self.orientation])
        f_name = os.path.join(os.path.dirname(__file__), 'car.urdf')
        self.car = p.loadURDF(fileName=f_name,
                              basePosition=[base[0], base[1], 0.1],
                              physicsClientId=client,
                              baseOrientation=self.orientation_quat)

        # Joint indices as found by p.getJointInfo()
        self.steering_joints = [3, 5]
        self.drive_joints = [4, 6, 7, 8]
        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 20

    def get_ids(self):
        return self.car, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional
        throttle, steering_angle = action[0], action[1]

        # Clip throttle and steering angle to reasonable values
        throttle = min(max(throttle, 1), 1.5)
        steering_angle = max(min(steering_angle, 135), -135)

        # Set the steering joint positions
        p.setJointMotorControlArray(self.car, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2,
                                    physicsClientId=self.client)

        # Calculate drag / mechanical resistance ourselves
        # Using velocity control, as torque control requires precise models
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/240 of a second
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
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))

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

        # Get the velocity of the car
        vel = p.getBaseVelocity(self.car, self.client)[0][0:2]

        dist_to_goal = math.sqrt(((pos[0] - goal[0]) ** 2 +
                                  (pos[1] - goal[1]) ** 2))

        # Create a dictionary for observation
        observation = {
            'image': rgb_image,
            'vector': pos + ori + vel + tuple([dist_to_goal])
        }

        return observation
