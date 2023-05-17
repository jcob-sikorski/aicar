import numpy as np
import driving
from custom_policy import CustomPolicy
from stable_baselines3 import PPO

# when car is going in the direction of a goal it correctly goes in that direction
# car doesn't turn left/right when nothing is in front of it
# car doesn't have a clear output where it should go
# car avoids all the boxes correctly
# when car sees the goal it correctly rides to the goal
# !!!!!!!!!! goal appears to little times - so car doesn't really know where it should align to


# TODO possibly include camera view
# TODO instead of mlp use CNN

# CustomPolicy is the name of the custom policy class you defined
model = PPO(CustomPolicy, "driving-v0", verbose=1, n_steps=10000, ent_coef=0.2)
model.learn(1)
# TODO tune the hyperparamters, rewards, experiment with ranges of angle and velocity
vec_env = model.get_env() # TODO stuck and power intensive when doing this
# mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
# print(mean_reward)

model.save("ppo_recurrent")
del model # remove to demonstrate saving and loading

model = PPO.load("ppo_recurrent")

obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones
    vec_env.render("human")