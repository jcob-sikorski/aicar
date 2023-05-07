# AI Car Navigation Challenge

This project simulates an AI-controlled car navigating through a randomly generated environment consisting of obstacles and a green rectangle representing the goal. The car is equipped with sensors that provide information about the distance to the obstacles, which is then used by an AI algorithm to control the car.

<img width="640" alt="img" src="https://user-images.githubusercontent.com/45105669/236634928-76af8f75-f135-45c9-95d7-2f9bc10fb377.png">

## Requirements

- Python 3.6 or later
- PyBullet physics engine
- OpenAI Gym

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