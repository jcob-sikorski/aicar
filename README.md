# AI Car Navigation Challenge

This project simulates an AI-controlled car navigating through a randomly generated environment consisting of obstacles and a green rectangle representing the goal. The car is equipped with sensors that provide information about the distance to the obstacles, which is then used by an AI algorithm to control the car.

<img width="1440" alt="Screen Shot 2023-05-16 at 6 04 07 PM" src="https://github.com/jcob-sikorski/aicar/assets/45105669/e8306068-c960-4a97-afeb-70da85ab423c">


## Requirements

- Python 3.6 or later
- PyBullet physics engine
- OpenAI Gym
- stable_baselines3
- numpy
- opencv

## Installation

Clone the repository:

https://github.com/jcob-sikorski/aicar.git

## Usage

1. To start the simulation: ```python main.py```
2. The AI-controlled car will begin navigating through the randomly generated environment, attempting to reach the green rectangle representing the goal.
3. The simulation will end when the car reaches the goal or collides with an obstacle.

## Customization

- `num_boxes`: number of randomly generated obstacles
- `box_size`: size of the obstacles (length, width, height)
- `goal_position`: position of the green rectangle representing the goal
