import gymnasium
import torch
from agent import TRPOAgent
import driving
import time


def main():
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    agent = TRPOAgent(policy=nn)

    agent.load_model("driving/agent.pth")
    agent.train("driving-v0", seed=0, batch_size=5000, iterations=200,
                max_episode_length=250, verbose=True)
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
