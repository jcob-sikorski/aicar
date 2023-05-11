import gymnasium
import torch
import torchvision
from agent import TRPOAgent
import driving
import time
from keras import Model
from keras.layers import Input, Dense, Bidirectional, LSTM


def main():
    # Observation: algorithm plays safe it doesn't move anywhere - meaning it learned not to hit the boxes but also that hitting a goal is too risky
    # TODO add asserts to test correctness of input data
    # TODO instead of LiDAR add positions of randomly generaetd environemnt on the fly
    # TODO generate new environment, pos of car and goal each time

    # TODO use instead MLP-LSTM policy from keras
    nn = torch.nn.Sequential(torch.nn.Linear(38, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))

    # TODO use PPO from pytorch
    agent = TRPOAgent(policy=nn)

    # agent.load_model("driving/agent.pth")
    agent.train("driving-v0", seed=0, batch_size=5000, iterations=1,
                max_episode_length=1000, verbose=True)
    agent.save_model("agent.pth")

    env = gymnasium.make('driving-v0')
    ob = env.reset()
    while True:
        action = agent(ob)
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)


if __name__ == '__main__':
    main()
