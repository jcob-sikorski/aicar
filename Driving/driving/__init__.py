from gymnasium.envs.registration import register
register(
    id='driving-v0',
    entry_point='driving.envs:DrivingEnv'
)