import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def same_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    env.reset(seed=seed)

def main():
    config = {
        "seed": 0,
    }
    env = gym.make("Hopper-v4", new_step_api=True)
    same_seed(config["seed"], env)

if __name__ == '__main__':
    main()
