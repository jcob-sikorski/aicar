import gymnasium as gym
import numpy as np
import math
import pybullet as p
from driving.resources.car import Car
from driving.resources.plane import Plane
from driving.resources.obstacle import Obstacle
from driving.resources.goal import Goal
import matplotlib.pyplot as plt


class DrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32))
        # TODO change observation space to include data from positions of boxes
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 5, 5, 10, 10], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def step(self, action):
        # Feed action to the car and get observation of a car's state
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        # Compute distance as L2 change in distance to a rayHit
        # dist_to_box = [math.sqrt(((car_ob[0] - rayHit[0]) ** 2 +
        #                           (car_ob[1] - rayHit[1]) ** 2)) for rayHit in car_ob[3:]]
        
        dist_to_box = [math.sqrt(((car_ob[0] - car_ob[i:i+3][0]) ** 2 +
                                  (car_ob[1] - car_ob[i:i+3][1]) ** 2))
                                  for i in range(3, len(car_ob), 3)]

        # TODO test a collision with a box

        # Compute reward as L2 change in distance to a goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        self.prev_dist_to_goal = dist_to_goal

        # Done by running off the boundaries
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or
                car_ob[1] >= 10 or car_ob[1] <= -10):
            self.done = True
            reward = -20
        # Done by reaching a goal
        elif dist_to_goal < 1:
            self.done = True
            reward = 100
        # Done by hitting a box
        elif any(num <= 0.3 for num in dist_to_box):
            self.done = True
            reward = -50

        ob = np.array(car_ob[:6] + tuple(self.goal) + car_ob[6:], dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # Reload the plane and car
        Plane(self.client)

        # TODO test randomizing the starting position of the player and the goal for each episode

        # TODO create a clear way to specify the environment - number of boxes and the seed
        # seed for environment creation
        np.random.seed(54)

        # Define the locations and dimensions of the buildings
        building_pos = np.random.uniform(low=-10, high=10, size=(10, 2))

        for b in building_pos[:-1]:
            self.done = False

            # Visual element of the goal
            Obstacle(self.client, (b[0], b[1]))

        # omit the seed
        rg = np.random.default_rng()

        # Generate a new goal position until it is different from all the building positions
        while True:
            # Generate a new random position for the goal
            self.goal = rg.uniform(low=-10, high=10, size=(1, 2))[0].astype(np.float32)

            if not any(math.sqrt((b_pos[0] - self.goal[0]) ** 2 + 
                        (b_pos[1] - self.goal[1]) ** 2) <= 1 for b_pos in building_pos):
                break

        car_pos = (0, 0)
        # Generate a new goal position until it is different from all the building positions
        while True:
            # Generate a new random position for the goal
            car_pos = rg.uniform(low=-10, high=10, size=(1, 2))[0]

            if not any(math.sqrt((b_pos[0] - car_pos[0]) ** 2 +
                    (b_pos[1] - car_pos[1]) ** 2) <= 1 for b_pos in building_pos):
                break

        self.car = Car(self.client, car_pos)

        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)

        # Get observation to return
        car_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))

        return np.array(car_ob[:6] + tuple(self.goal) + car_ob[6:], dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, -1, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 3]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)
