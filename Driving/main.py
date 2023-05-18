# import numpy as np
# import driving
# from custom_policy import CustomPolicy
# from stable_baselines3 import PPO

# # when car is going in the direction of a goal it correctly goes in that direction
# # car doesn't turn left/right when nothing is in front of it
# # car doesn't have a clear output where it should go
# # car avo
# ids all the boxes correctly
# # when car sees the goal it correctly rides to the goal
# # !!!!!!!!!! goal appears to little times - so car doesn't really know where it should align to


# TODO try to use only the image as the observation - cnnlstmrecurrentPPO policy for image only
# TODO rewards based only on green, white and blue colours

# TODO possibly include camera view
# TODO instead of mlp use CNN
# TODO tune the hyperparamters, rewards, experiment with ranges of angle and velocity

import numpy as np
import driving
from sb3_contrib import RecurrentPPO

model = RecurrentPPO("MultiInputLstmPolicy", "driving-v0", verbose=1, ent_coef=0.2, n_steps=128, learning_rate=0.003, seed=42)
model.learn(total_timesteps=128)

env = model.get_env()

# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)

observation = env.reset()
while True:
    action, lstm_states = model.predict(observation, state=lstm_states, episode_start=episode_starts, deterministic=True)
    observation, rewards, dones, info = env.step(action)
    episode_starts = dones
    env.render()

    if episode_starts:
        dones=False
        episode_starts=False
        env.reset()