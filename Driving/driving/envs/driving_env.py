# Import the required libraries and modules
import gym
import numpy as np
import math
import pybullet as p
import matplotlib.pyplot as plt
import cv2
from gym.spaces import Box
from driving.resources.car import Car
from driving.resources.plane import Plane
from driving.resources.obstacle import Obstacle
from driving.resources.goal import Goal
import pygame

class DrivingEnv(gym.Env):
    """
    Driving environment for a Gym-styled reinforcement learning environment.
    This environment simulates a car driving experience with obstacles.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Initialize the driving environment.
        """
        # Action space of the car consists of throttle and steering angle
        # Note: Check the bounds of throttle and angle
        self.action_space = Box(
            low=np.array([1, -135], dtype=np.float32),
            high=np.array([1.5, 135], dtype=np.float32))

        # Observation space of the car consists of RGB image
        img_height = 60
        img_width = 80
        img_channels = 3
        self.observation_space = Box(low=0, high=255, 
                                     shape=(img_height, img_width, img_channels), 
                                     dtype=np.uint8)

        self.np_random, _ = gym.utils.seeding.np_random()

        # Initialize PyBullet client
        self.client = p.connect(p.DIRECT)

        # Set the timestep to reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        # Initialize car and goal
        self.car = None
        self.goal = None
        self.obstacles = []
        self.done = False
        self.prev_dist_to_goal = 0
        self.timestep = 0

        # Initialize reward tracking
        self.reward_tracking = {
            'turning': 0,
            'obstacle\navoidance': 0,
            'green\npixels': 0,
            'boundary\npenalty': 0,
            'goal\nreward': 0,
            'collision\npenalty': 0,
            'blue\npixels': 0,
            'distance\npenalty':0
        }

        # Initialize figure for rendering environment
        self.fig, self.axs = plt.subplots(1, 2)  # Create a 1x2 subplot grid
        self.rendered_img = None
        self.reward_bar = None
        self.reward_types = list(self.reward_tracking.keys())
        self.reward_bar_heights = np.zeros(len(self.reward_types))

        self.image_zeros = np.zeros((100, 100, 4))

        # Initialize Pygame and create a window
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))


    def plot_rewards(self):
        """
        Plot the rewards during training.
        """
        if self.reward_bar is None:
            self.reward_bar = self.axs[1].bar(self.reward_types, self.reward_bar_heights)
            self.axs[1].set_yscale('symlog')  # Set symlog scale with base 10
            self.axs[1].set_xticklabels(self.reward_types, fontsize=8)  # Increase fontsize here
            self.fig.tight_layout()
        else:
            for i, reward_type in enumerate(self.reward_types):
                self.reward_bar_heights[i] = self.reward_tracking[reward_type]
            for bar, height in zip(self.reward_bar, self.reward_bar_heights):
                bar.set_height(height)
            self.axs[1].relim()
            self.axs[1].autoscale_view()


    def step(self, action):
        """
        Perform an action in the environment and return the next state, the reward and whether the state is terminal.

        Parameters
        ----------
        action : np.array
            Action to be performed.

        Returns
        -------
        image : np.array
            The observation after performing the action.
        reward : float
            The reward achieved by performing the action.
        self.done : bool
            Whether the episode has ended.
        info : dict
            Extra information, currently empty.
        """

        # Feed the action to the car and step the simulation
        self.car.apply_action(action)
        p.stepSimulation()

        # Retrieve the car's position and orientation from the simulation
        car_id, client_id = self.car.get_ids()
        car_pos, ang = p.getBasePositionAndOrientation(car_id, client_id)
        ang = p.getEulerFromQuaternion(ang)

        # Compute the distance to the goal
        dist_to_goal = np.linalg.norm(self.goal - car_pos[:2])

        # Obtain the observation image
        image = self.car.get_observation(self.goal)

        # Initialize reward
        reward = 0

        # Penalize the car if it is moving away from the goal
        if dist_to_goal > self.prev_dist_to_goal:
            distance_penalty = (dist_to_goal - self.prev_dist_to_goal) * 500  # Can adjust the scaling factor
            reward -= distance_penalty
            self.reward_tracking['distance\npenalty'] = -distance_penalty

            # Reward for turning
            turning_reward = abs(action[1]) * 20
            reward += turning_reward
            self.reward_tracking['turning'] = turning_reward
        else:
            self.reward_tracking['distance\npenalty'] = 0

        self.prev_dist_to_goal = dist_to_goal

        # Define the color bounds
        lower_blue = np.array([0, 0, 120])
        upper_blue = np.array([100, 100, 255])

        # Create a mask that only contains the blue range
        mask = cv2.inRange(image, lower_blue, upper_blue)

        # Compute the ratio of blue pixels to the total pixels
        ratio_blue = cv2.countNonZero(mask) / (image.size / 3)
        reward -= ratio_blue * 50
        self.reward_tracking['blue\npixels'] = -ratio_blue * 60

        if ratio_blue > 100:
            # Small reward for turning
            turning_reward = abs(action[1]) * 100
            reward += turning_reward
            self.reward_tracking['turning'] = turning_reward

        # Add obstacle avoidance reward
        for obstacle in self.obstacles[:-1]:
            obstacle_pos = np.array(obstacle.get_pos())
            dist_to_obstacle = np.linalg.norm(car_pos[:2] - obstacle_pos)
            reward -= (1 / dist_to_obstacle)*10
            self.reward_tracking['obstacle\navoidance'] = -1 / dist_to_obstacle*10

        # Define color bounds for green
        lower_green = np.array([0, 120, 0])
        upper_green = np.array([100, 255, 100])

        # Create a mask that only contains the green range
        mask = cv2.inRange(image, lower_green, upper_green)

        # Compute the ratio of green pixels to the total pixels
        ratio_green = cv2.countNonZero(mask) / (image.size / 3)
        reward += ratio_green * 1000
        self.reward_tracking['green\npixels'] = ratio_green * 1000

        # Check if episode has ended
        if (car_pos[0] >= 15 or car_pos[0] <= -5 or car_pos[1] >= 15 or car_pos[1] <= -5):
            self.done = True  # Episode ends if car is off the boundaries
        elif dist_to_goal < 1:
            self.done = True  # Episode ends if car has reached the goal
        else:
            # Check for collision with obstacles
            car_id = self.car.get_ids()[0]
            for obstacle in self.obstacles[:-1]:
                if p.getContactPoints(car_id, obstacle.get_id(), physicsClientId=self.client):
                    self.done = True  # Episode ends if car hits an obstacle
                    break

        self.timestep += 1
        return image, reward, self.done, dict()
    

    def seed(self, seed=None):
        """
        Set the seed for this environment's random number generator.

        Parameters
        ----------
        seed : int, optional
            The seed for the random number generator.

        Returns
        -------
        list(int)
            List of seeds used in this env's random number generators.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation.

        Returns
        -------
        np.array
            The initial observation of the space.
        """
        # Reset state variables
        self.done = False
        self.obstacles = []

        # Reset the simulation and set gravity
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # Reload the plane and car
        Plane(self.client)

        # Reset environment creation seed
        np.random.seed(54)

        # Define the locations and dimensions of the buildings
        building_pos = np.random.uniform(low=0, high=10, size=(10, 2))

        # Create visual elements for the goal and obstacles
        for b in building_pos[:-1]:
            # Visual element of the goal
            self.obstacles.append(Obstacle(self.client, (b[0], b[1])))

        # Omit the seed
        rg = np.random.default_rng()

        # Generate a valid position for the goal
        while True:
            self.goal = rg.uniform(low=0, high=10, size=(1, 2))[0].astype(np.float32)
            if not any(np.linalg.norm(b_pos - self.goal) <= 1 
                       for b_pos in building_pos):
                break

        # Generate a valid position for the car
        while True:
            car_pos = rg.uniform(low=0, high=10, size=(1, 2))[0]
            if not any(np.linalg.norm(b_pos - car_pos) <= 1 
                       for b_pos in building_pos):
                break

        # Compute the angle to the goal and create the car with this orientation
        goal_vec = self.goal - car_pos
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        self.car = Car(self.client, car_pos, goal_angle)

        # Visual element of the goal
        Goal(self.client, self.goal)

        # Return the initial observation
        return self.car.get_observation(self.goal)


    def render(self, mode='human'):        
        """
        Render the environment.
    
        Parameters
        ----------
        mode : str, optional
            The mode to use for rendering.
        """
        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1, nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2
    
        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1])) 
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec) 
    
        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert image to RGB format for pygame
    
        # Create pygame surface
        pygame_frame = pygame.image.fromstring(frame.tostring(), frame.shape[:2][::-1], 'RGB')
    
        # Scale up the frame to desired size, let's say 500x500 pixels
        pygame_frame = pygame.transform.scale(pygame_frame, (500, 500))
    
        # Assuming self.screen has been initialized with matching size
        # self.screen = pygame.display.set_mode((500, 500))
    
        self.screen.blit(pygame_frame, (0, 0))
        pygame.display.flip()  # Update the full display surface to the screen


    def close(self):
        """
        Clean up the environment's resources.
        """
        p.disconnect(self.client) # Disconnect the PyBullet client
        pygame.quit() # Quit Pygame 