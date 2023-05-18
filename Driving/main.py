# Import the required libraries and modules
import numpy as np
import driving
from sb3_contrib import RecurrentPPO

# TODO how to end the episode when car falls off the ground?
# TODO minimizize pygame window and make the screen the priority in macos
# TODO model is circling around
# TODO model is turning only to the left

def main():
    # Initialize a Recurrent Proximal Policy Optimization model with a LSTM Policy
    # for the 'driving-v0' environment. The model is configured with the provided
    # hyperparameters and a fixed seed for reproducibility.
    model = RecurrentPPO(
        "CnnLstmPolicy",   # The policy model to use - a Convolutional Neural Network with a LSTM
        "driving-v0",      # The custom driving environment
        verbose=1,         # Verbosity level for logging
        ent_coef=0.2,      # Entropy coefficient for the loss calculation
        n_steps=256,       # The number of steps to run for each environment per update
        batch_size=64,
        learning_rate=0.003, # The learning rate
        seed=42            # Random seed for reproducibility
    )

    # Train the model for a specified number of timesteps
    model.learn(total_timesteps=1000)

    # Get the environment from the model
    env = model.get_env()

    # Initialize the LSTM states
    # cell and hidden state of the LSTM
    lstm_states = None

    # Initialize the number of environments
    num_envs = 1

    # Initialize the episode starts signal, used to reset the LSTM states
    episode_starts = np.ones((num_envs,), dtype=bool)

    # Reset the environment and get the initial observation
    observation = env.reset()

    timestep = 0
    # Start an infinite loop for playing the game
    while True:
        timestep += 1
        if timestep > 500:
            timestep = 0
            env.reset()
            done = False

        # Get the model's action based on the current observation, LSTM state, and episode start signal
        action, lstm_states = model.predict(observation, state=lstm_states, episode_start=episode_starts, deterministic=True)

        # Take the action in the environment and get the new observation, reward, done signal, and additional info
        observation, _, done, _ = env.step(action)

        # Render the environment
        env.render()

        # Check if the episode has ended
        if done:
            # Reset the done signal and the episode starts signal
            done = False
            timestep = 0
            # Reset the environment
            env.reset()

if __name__ == "__main__":
    main()