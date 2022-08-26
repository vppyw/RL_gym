import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class Buffer():
    def __init__(self, max_len):
        self.arr = []
        self.max_len = max_len

    def store(self, state, nxt_state, reward, done, act):
        if len(self.arr) > self.max_len:
            self.arr = self.arr[1:]
        self.arr.append((state,
                         nxt_state,
                         reward,
                         done,
                         act))

    def sample(self, batch_size, device):
        idx = random.choices(range(len(self.arr)), k=batch_size)
        states = torch.from_numpy(np.vstack([
                                    self.arr[i][0] for i in idx
                                    ])).float().to(device)
        nxt_states = torch.from_numpy(np.vstack([
                                    self.arr[i][1] for i in idx
                                    ])).float().to(device)
        rewards = torch.from_numpy(np.vstack([
                                    self.arr[i][2] for i in idx
                                    ])).float().to(device)
        dones = torch.from_numpy(np.vstack([
                                    self.arr[i][3] for i in idx
                                    ])).float().to(device)
        acts = torch.from_numpy(np.vstack([
                                    self.arr[i][4] for i in idx
                                    ])).float().to(device)
        return states, nxt_states, rewards, dones, acts
        
    def __len__(self):
        return len(self.arr)

class QNet(nn.Module):
    def __init__(self, in_size, out_size, emb_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, out_size),
        )
        
    def forward(self, state):
        return self.fc(state)

class Agent():
    def __init__(self, state_size, act_size, config):
        self.state_size = state_size
        self.act_size = act_size

        self.qnet = QNet(state_size,
                            act_size,
                            config['seed']).to(config['device'])
        self.qnet_target = QNet(state_size,
                                    act_size,
                                    config['seed']).to(config['device'])
        self.opt = torch.optim.AdamW(self.qnet.parameters(),
                                        lr=config['lr'])

        self.buff = Buffer(config['buffer_size'])
        self.batch_size = config['batch_size']
        self.t_step = 0
        self.gamma = config['gamma']
        self.learn_step = config['learn_step']
        self.t = config['t']
        self.update_step = config['update_step']

        self.device = config['device']

    def act(self, state, eps=0.):
        if random.random() > eps:
            state = torch.from_numpy(state)\
                        .float()\
                        .unsqueeze(0)\
                        .to(self.device)
            self.qnet.eval()
            with torch.no_grad():
                act_vals = self.qnet(state)
            return np.argmax(act_vals.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.act_size))

    def step(self, state, nxt_state, reward, done, act):
        self.buff.store(state, nxt_state, reward, done, act)
        self.t_step += 1
        self.t_step %= self.learn_step * self.update_step
        if self.t_step % self.learn_step == 0:
            self.learn()
        if self.t_step % self.update_step == 0:
            self.soft_update()

    def learn(self):
        states, nxt_states, rewards, dones, acts = \
                                        self.buff.sample(self.batch_size,
                                                         self.device)
        self.qnet_target.eval()
        # TODO: fix shape bug
        q_nxt_states = self.qnet_target(nxt_states)\
                                    .detach()\
                                    .max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * q_nxt_states * (1 - dones)
        self.qnet.train()
        q_exps = self.qnet(states).gather(1, acts)
        loss = F.mse(q_exps, q_targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def soft_update(self, t=1.):
        for param, param_target in zip(self.qnet.parameters(),
                                        self.qnet_target.parameters()):
            param_target.data.copy_(self.t * param.data\
                                    + (1 - self.t) * param_target.data)

    def to(self, device):
        self.device = device
        self.q_net = self.q_net.to(device)
        self.q_net_target = self.q_net_target.to(device)

def same_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    env.seed(seed)

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
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    env = gym.make("LunarLander-v2")
    same_seed(config['seed'], env)
    
    agent = Agent(env.observation_space.shape[0],
                    env.action_space.n,
                    config)
    
    eps = config['eps_max']
    scores = []
    for episode in range(config['num_episode']):
        state = env.reset()
        score = 0
        for step in range(config['max_step']):
            act = agent.act(state, eps)
            nxt_state, reward, done, _ = env.step(act)
            agent.step(state, nxt_state, reward, done, act)
            break
            state = nxt_state
            score += reward
            if done:
                break

        scores.append(score)
        eps = max(config['eps_min'], eps * config['eps_decay'])

if __name__ == '__main__':
    main()
