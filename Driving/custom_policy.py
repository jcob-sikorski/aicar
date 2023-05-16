import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space["image"].shape[0]
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )
        # TODO LSTM?
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space["image"].sample()[None]).float()
            ).shape[1]

        self.linear = th.nn.Sequential(
            th.nn.Linear(n_flatten + 8, features_dim),  # +4 for the vector part
            th.nn.ReLU(),
        )

    def forward(self, observations):
        image = observations["image"]
        vector = observations["vector"]
        return self.linear(th.cat([self.cnn(image), vector], dim=1))


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=128))

    def _predict(self, observation: th.Tensor, deterministic: bool = False):
        latent_pi, _, latent_sde = self._get_latents(observation)
        distribution = self._get_action_dist_from_latents(latent_pi, latent_sde)
        
        if deterministic:
            actions = distribution.get_mean()
        else:
            actions = distribution.sample()

        # Ensure throttle is at least 1
        actions[0] = th.clamp(actions[0], min=1)
        
        return actions