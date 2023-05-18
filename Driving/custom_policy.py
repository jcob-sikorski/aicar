import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym import spaces

# TODO understand how models should be written
# uderstand each function

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True)

        self.linear = nn.Linear(64 * 80 * 60 + 16, 256)  # Adjust dimensions accordingly

    def forward(self, observations) -> th.Tensor:
        # if len(observations['image'].shape) == 4:
        print("#######################################")
        print(2)
        print(observations['image'].shape)
        print(observations['vector'].shape)
        print(len(observations['image'].shape))

        image = (observations['image'].float() / 255.0)
        vector = observations['vector']
        # Reshape image tensor to (1, 3, 240, 320)
        if len(image.shape) == 3:
            image = image.unsqueeze(0).permute(0, 3, 1, 2)
    
        # Reshape vector tensor to (1, 8)
        if len(vector.shape) == 1:
            vector = vector.unsqueeze(0)

        # TODO the bug:
        # RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input
        # [1, 240, 320, 3] to have 3 channels, but got 240 channels instead
        image = self.cnn(image).view(image.size(0), -1)

        # Expand the vector tensor to match the batch size of the image tensor
        vector = vector.expand(image.size(0), -1, -1)


        vector, lstm_hidden_state = self.lstm(vector)
        vector = vector[:, -1, :]

        print(image.shape)
        print(vector.shape)
        print("#######################################")
        return self.linear(th.cat((image, vector), dim=1)), lstm_hidden_state


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, features_extractor_class=CustomCombinedExtractor, 
                                           features_extractor_kwargs=dict())
        self.lstm_hidden_state = None

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        obs_image = obs['image']
        obs_vector = obs['vector'].unsqueeze(0)

        latent_pi, latent_vf, lstm_hidden_state = self._get_latents(obs_image, obs_vector)
        self.lstm_hidden_state = lstm_hidden_state

        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution

    def _get_latents(self, obs_image: th.Tensor, obs_vector: th.Tensor):
        features, lstm_hidden_state = self.features_extractor({'image': obs_image, 'vector': obs_vector})
        latent_pi, latent_vf = self.mlp_extractor(features)
        return latent_pi, latent_vf, lstm_hidden_state

    def predict(self, observation, state=None, mask=None, deterministic=False):
        with th.no_grad():
            obs_tensor = th.as_tensor(observation).to(self.device)
            distribution = self.forward(obs_tensor, deterministic)
            action = distribution.get_actions()
            action = th.clamp(action, min=1)

        return action.cpu().numpy(), self.lstm_hidden_state


    

# if image and the vector in observartion has such shape:
# torch.Size([240, 320, 3])
# torch.Size([8])
# make them this shape:
# torch.Size([1, 3, 240, 320])
# torch.Size([1, 8])