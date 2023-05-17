import gym
import numpy as np
import math
import pybullet as p
from driving.resources.car import Car
from driving.resources.plane import Plane
from driving.resources.obstacle import Obstacle
from driving.resources.goal import Goal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gym.spaces import Dict, Box
import cv2
import time
from IPython.display import clear_output

# TODO add parallelism to train faster the model

# 1. Reward turning actions: You could give a small positive reward for turning actions. 
# This will encourage the car to explore turning actions more often. 
# However, be careful to balance this correctly; if the reward for turning is too high, 
# the car might just spin in circles.

# 2. Penalize straight-line driving: Similar to the above, but instead of rewarding turns, 
# you penalize driving straight. This will encourage the car to explore and
#  learn how to navigate with turns more often. Be careful with the balance here 
# too; if the penalty is too high, the car might avoid driving straight even 
# when it's the best action.

# 3. Add rewards based on the angle to the goal: Instead of just rewarding
# based on distance to the goal, you could also add a reward based on the angle 
# to the goal. If the car is facing the goal directly, it gets a higher reward.
#  This will encourage the car to turn towards the goal more often.

# 4. Add rewards based on obstacle avoidance: You could also add a reward 
# based on how well the car avoids obstacles. For example, if the car turns to 
# avoid an obstacle, it gets a reward. This would encourage the car to 
# turn more often, especially in situations where there are obstacles.

# 5. You could modify the reward function in your step method to include a 
# reward for how much green the car sees in the current observation. 
# This would encourage the car to drive towards the green goal:

# This will provide the agent with a higher reward 
# the more green it sees, thereby encouraging it to drive towards 
# the green box. Note that the range for the green color in the 
# calculate_green function might need to be adjusted based on the 
# actual color values of the green box in your environment. 
# Also, it's important to remember that the color range is defined 
# in BGR format, not RGB.



class DrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = Box(
            low=np.array([1, -135], dtype=np.float32),
            high=np.array([1.5, 135], dtype=np.float32))
        
        img_height = 240
        img_width = 320
        img_channels = 3
        img_space = Box(low=0, high=255, shape=(img_height, img_width, img_channels), dtype=np.uint8)
        
        vector_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.observation_space = Dict({
            'image': img_space,
            'vector': vector_space
        })
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.goal = None
        self.obstacles = []
        self.done = False
        self.prev_dist_to_goal = 0
        self.rendered_img = None
        self.render_rot_matrix = None
        self.timestep = 0
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
        # self.reset()
        self.fig, self.axs = plt.subplots(1, 2)  # Create a 1x2 subplot grid
        self.rendered_img = None
        self.reward_bar = None
        self.reward_types = list(self.reward_tracking.keys())
        self.reward_bar_heights = np.zeros(len(self.reward_types))


    def plot_rewards(self):
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
        # Feed action to the car and get observation of a car's state
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation(self.goal)

        reward = 0

        # Compute reward as L2 change in distance to a goal
        dist_to_goal = car_ob['vector'][7]
        # Add a penalty if the car is getting further away from the goal
        if dist_to_goal > self.prev_dist_to_goal:
            distance_penalty = (dist_to_goal - self.prev_dist_to_goal) * 500  # You can adjust the scaling factor
            reward -= distance_penalty
            self.reward_tracking['distance\npenalty'] = -distance_penalty
            turning_reward = abs(action[1]) * 20
            reward += turning_reward # small reward for turning
            self.reward_tracking['turning'] = turning_reward
        else:
            self.reward_tracking['distance\npenalty'] = 0

        self.prev_dist_to_goal = dist_to_goal
        car_pos = car_ob['vector'][:2]

        # TODO examine range of rewards some might be to big or too small!
        # Observations: the car isn't suffietnly avoiding the boxes.
        # When car has the clear way to the box it corrctly goes to it
        # When car has the boxes in the way to the goal it tries to avoid them but not sufficiently.
        # Reward for turning

        # TODO the model should not be penalized for driving into a green box

        # TODO for avoiding the obtacles we can use the same logic for blue boxes as for green box
        # Define lower and upper bounds for the green color
        # Define lower and upper bounds for the blue color
        lower_blue = np.array([0, 0, 120])
        upper_blue = np.array([100, 100, 255])

        # TODO add a penalty for being more far away from the goal than previously it should be bigger than turning reward

        # Create a mask that only allows the blue range
        mask = cv2.inRange(car_ob['image'], lower_blue, upper_blue)

        # Calculate the ratio of blue pixels to total pixels
        ratio_blue = cv2.countNonZero(mask) / (car_ob['image'].size / 3)
        reward -= ratio_blue * 50
        self.reward_tracking['blue\npixels'] = -ratio_blue * 60
        if ratio_blue > 100:
            turning_reward = abs(action[1]) * 100
            reward += turning_reward # small reward for turning
            self.reward_tracking['turning'] = turning_reward

        # Add obstacle avoidance reward
        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle.get_pos())
            dist_to_obstacle = np.linalg.norm(car_pos - obstacle_pos)
            if dist_to_obstacle < 1:
                reward -= (1 / dist_to_obstacle)*10  # You can adjust the scaling factor if needed
                self.reward_tracking['obstacle\navoidance'] = -1 / dist_to_obstacle*10 if dist_to_obstacle < 1 else 0

        # Define lower and upper bounds for the green color
        lower_green = np.array([0, 120, 0])
        upper_green = np.array([100, 255, 100])

        # Create a mask that only allows the green range
        mask = cv2.inRange(car_ob['image'], lower_green, upper_green)
        
        # Calculate the ratio of green pixels to total pixels
        ratio_green = cv2.countNonZero(mask) / (car_ob['image'].size / 3)
        reward += ratio_green * 1000
        self.reward_tracking['green\npixels'] = ratio_green * 1000

        # Done by running off the boundaries
        if (car_ob['vector'][0] >= 15 or car_ob['vector'][0] <= -5 or
                car_ob['vector'][1] >= 15 or car_ob['vector'][1] <= -5):
            self.done = True
            reward = -50
            self.reward_tracking['boundary\npenalty'] = -50
        # Done by reaching a goal
        elif dist_to_goal < 1:
            self.done = True
            reward = 100
            self.reward_tracking['goal\nreward'] = 100
        # Done by hitting a box
        car_id = self.car.get_ids()[0]
        for obstacle in self.obstacles:
            if p.getContactPoints(car_id, obstacle.get_id(), physicsClientId=self.client):
                self.done = True
                reward = -50
                break
            self.reward_tracking['collision\npenalty'] = -50 if self.done else 0

        self.timestep += 1
        print(self.timestep, reward)
        # self.render()
        # self.plot_rewards()
        # plt.pause(0.0001)  # you can adjust this value as needed
        return car_ob, reward, self.done, dict()
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.done = False
        self.obstacles = []

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # Reload the plane and car
        Plane(self.client)

        # TODO create a clear way to specify the environment - number of boxes and the seed
        # seed for environment creation
        np.random.seed(54)

        # Define the locations and dimensions of the buildings
        building_pos = np.random.uniform(low=0, high=10, size=(10, 2))
        # TODO does whole simulation needs to be reseted or only the position of the goal and the car may be changed
        for b in building_pos[:-1]:
            # Visual element of the goal
            self.obstacles.append(Obstacle(self.client, (b[0], b[1])))

        # omit the seed
        rg = np.random.default_rng()

        # Generate a new goal position until it is different from all the building positions
        while True:
            # Generate a new random position for the goal
            self.goal = rg.uniform(low=0, high=10, size=(1, 2))[0].astype(np.float32)
            # self.goal = np.random.randint(0, 10, size=2)

            if not any(math.sqrt((b_pos[0] - self.goal[0]) ** 2 + 
                        (b_pos[1] - self.goal[1]) ** 2) <= 1 for b_pos in building_pos):
                break

        car_pos = (0, 0)
        # Generate a new car position until it is different from all the building positions
        while True:
            # Generate a new random position for the car
            car_pos = rg.uniform(low=0, high=10, size=(1, 2))[0]
            # car_pos = np.random.randint(-10, 10, size=2)

            if not any(math.sqrt((b_pos[0] - car_pos[0]) ** 2 +
                    (b_pos[1] - car_pos[1]) ** 2) <= 1 for b_pos in building_pos):
                break

        # Compute the angle to the goal
        goal_vec = self.goal - car_pos
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])

        # Create the car with the computed orientation
        self.car = Car(self.client, car_pos, goal_angle)

        # self.car = Car(self.client, car_pos)

        # Visual element of the goal
        Goal(self.client, self.goal)

        # Get observation to return
        return self.car.get_observation(self.goal)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = self.axs[0].imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        self.axs[0].draw_artist(self.rendered_img)
        self.fig.canvas.blit(self.axs[0].bbox)

    def close(self):
        p.disconnect(self.client)
