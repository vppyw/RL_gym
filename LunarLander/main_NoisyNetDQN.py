import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from PIL import Image
from tqdm import tqdm

import model

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
        'seed': 0,
        'num_episode': 1000,
        'max_step': 1000,
        'batch_size': 64,
        'buffer_size': 1e5,
        'lr': 1e-4,
        'learn_step': 1,
        'update_step': 4,
        'gamma': 0.99,
        't': 1e-3,
        'eps_max': 1.0,
        'eps_min': 0.005,
        'eps_decay': 0.999,
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
        'log': 'noisynet_dqn_scores.csv',
        'model': 'NoisyNetDQN.pt',
    }
    env = gym.make("LunarLander-v2", new_step_api=True)
    same_seed(config['seed'], env)

    agent = model.NoisyNetAgent(env.observation_space.shape[0],
                    env.action_space.n,
                    config)
    
    eps = config['eps_max']
    scores = []
    for episode in tqdm(range(config['num_episode']), ncols=50):
        state, _ = env.reset()
        score = 0
        agent.set_noise()
        for step in range(config['max_step']):
            act = agent.act(state, noise=True)
            nxt_state, reward, done, _, _ = env.step(act)
            agent.step(state, nxt_state, reward, done, act)
            state = nxt_state
            score += reward
            if done:
                break
        scores.append(score)

    agent.qnet.eval()
    agent.qnet.to('cpu')
    torch.save(agent.qnet.state_dict(), config['model'])
    with open(config['log'], 'w') as f:
        f.write('episodes,score\n')
        for idx, val in enumerate(scores):
            f.write(f'{idx},{val}\n')

if __name__ == '__main__':
    main()
