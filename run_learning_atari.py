import gym
from utils.atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch

env_ids = ["PongNoFrameskip-v4",
           "SeaquestNoFrameskip-v4",
           "MsPacmanNoFrameskip-v4"]

env_id = env_ids[-1]
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

state = env.reset()
print(state.shape)
