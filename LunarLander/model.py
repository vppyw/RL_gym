import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from multiprocessing import Pool

class Buffer():
    """
    Buffer with random smaple
    """
    def __init__(self, max_len):
        self.arr = []
        self.max_len = max_len

    def store(self, state, nxt_state, reward, done, act):
        """
        Add new experience to buffer
        """
        if len(self.arr) > self.max_len:
            self.arr = self.arr[1:]
        self.arr.append((state,
                         nxt_state,
                         reward,
                         done,
                         act))

    def sample(self, batch_size, device):
        """
        Random sample batch size of experiences
        output: states, nxt_states, rewards, dones, acts, idx
        """
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
        return states, nxt_states, rewards, dones, acts, idx
        
    def __len__(self):
        """
        Return the size of buffer
        """
        return len(self.arr)

class PrioritizedBuffer():
    def __init__(self, max_len):
        """
        Prioritized Replay implemented by sum-tree
        """
        self.max_len = max_len
        self.arr = []
        self.losses = []
        self.weights = []

    def store(self, state, nxt_state, reward, done, act, loss=1e-12):
        """
        Add new experience to buffer with given weight
        """
        if len(self.arr) > self.max_len:
            self.arr = self.arr[1:]
        self.arr.append((state,
                         nxt_state,
                         reward,
                         done,
                         act))
        self.losses.append(loss)
        self.weights.append(loss) # Set weight same as loss
        
    def update_loss(self, idxes, losses):
        if len(idxes) != len(losses):
            raise ValueError

        for idx, loss in zip(idxes, losses):
            self.losses[idx] = loss.item()
            self.weights[idx] = loss.item() # Set weight same as loss

    def sample(self, batch_size, device):
        """
        Random sample batch size of experiences
        output: states, nxt_states, rewards, dones, acts, idx
        """
        idx = random.choices(range(len(self.weights)), weights=self.weights, k=batch_size)
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
        return states, nxt_states, rewards, dones, acts, idx
        
    def __len__(self):
        return len(self.arr)
    
class QNet(nn.Module):
    """
    Basic network for Q-learning
    intput: (batch size, in_size)
    output: (batch size, out_size)
    """
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

class DuelingQNet(nn.Module):
    """
    Network for Dueling DQN
    """
    def __init__(self, in_size, out_size, emb_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        self.v_fc = nn.Sequential(
            nn.Linear(emb_size, 1),
        )
        self.a_fc = nn.Sequential(
            nn.Linear(emb_size, out_size)
        )

    def forward(self, state):
        comm = self.fc(state)
        v = self.v_fc(comm)
        a = self.a_fc(comm)
        # contraints
        a_norm = a - a.mean(dim=1, keepdim=True)
        return v + a_norm

class NoisyLayer(nn.Module):
    """
    Noisy Linear Layer
    """
    def __init__(self, in_features, out_features, var_init):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.var_init = var_init
        self.weights_mu = nn.Parameter(
                            torch.FloatTensor(out_features, in_features)
                          )
        self.weights_sig = nn.Parameter(
                            torch.FloatTensor(out_features, in_features)
                           )
        self.register_buffer('weights_eps',
                             torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sig = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_eps', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.set_noise()

    def forward(self, x):
        if self.training:
            weights = self.weights_mu + self.weights_sig * self.weights_eps
            bias = self.bias_mu + self.bias_sig * self.bias_eps
        else:
            weights = self.weights_mu
            bias = self.bias_mu
        return F.linear(x, weights, bias)
    
    def reset_parameters(self):
        """
        Initialisation parameters for factorised noisy network
        """
        self.set_mu(self.weights_mu)
        self.set_sig(self.weights_sig)
        self.set_mu(self.bias_mu)
        self.set_sig(self.bias_sig)

    def set_mu(self, mu):
        bound = 1 / math.sqrt(self.in_features)
        mu.data.uniform_(-bound, bound)

    def set_sig(self, sig):
        sig.data.fill_(self.var_init / math.sqrt(self.in_features))
    def set_noise(self):
        """
        Setting new noise using factorised Gaussian noise
        """
        # Factorised Gaussian noise
        eps_i = self.f(self.in_features)
        eps_j = self.f(self.out_features)
        self.weights_eps.data.copy_(esp_j.outer(esp_i))

        # Problem: should the noise be the same as esp_j?
        esp_j = self.f(self.out_features)
        self.bias_eps.data.copy_(esp_j)

    def f(self, size):
        """
        for Factorised Gaussian noise
        """
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class NoisyNet(nn.Module):
    def __init__(self, in_size, out_size, emb_size=64):
        super().__init__()
        self.linear = nn.Linear(in_size, emb_size)
        self.noisy0 = NoisyLayer(emb_size, emb_size)
        self.noisy1 = NoisyLayer(emb_size, emb_size)
        
    def forward(self, state):
        x = F.relu(self.linear(state))
        x = F.relu(self.noisy0(x))
        x = self.noisy1(x)
        return x

    def set_noise(self):
        self.noisy0.set_noise()
        self.noisy1.set_noise()

class DuelingNoisyNet(nn.Module):
    def __init__(self, in_size, out_size, emb_size=64):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_size, emb_size)
            nn.ReLU(),
        )
        self.noisy = nn.NoisyLayer(emb_size, emb_size)
        self.v_noisy = nn.NoisyLayer(emb_size, 1)
        self.a_noisy = nn.NoisyLayer(emb_size, out_size)

    def forward(self, state):
        comm = self.linear(state)
        comm = F.relu(self.noisy(comm))
        v = self.v_noisy(comm)
        a = self.a_noisy(comm)
        a_norm = a - a.mean(dim=1, keepdim=True)
        return v + a

    def set_noise(self):
        self.noisy.set_noise()
        self.v_noisy.set_noise()
        self.a_noisy.set_noise()

class DQN_Agent():
    """
    Agent for DQN
    """
    def __init__(self, state_size, act_size, config):
        self.state_size = state_size
        self.act_size = act_size
        self.buff = Buffer(config['buffer_size'])
        self.batch_size = config['batch_size']
        self.t_step = 0
        self.gamma = config['gamma']
        self.learn_step = config['learn_step']
        self.t = config['t']
        self.update_step = config['update_step']
        self.device = config['device']
        self.prepare_net()

    def prepare_net(self):
        self.qnet = QNet(self.state_size,
                         self.act_size).to(config['device'])
        self.qnet_target = QNet(self.state_size,
                                self.act_size).to(config['device'])
        self.opt = torch.optim.AdamW(self.qnet.parameters(),
                                     lr=config['lr'])

    def act(self, state, eps=0.):
        """
        Actor with Epsilon Greedy
        """
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
        """
        Update for each step
        """
        self.buff.store(state, nxt_state, reward, done, act)
        self.t_step += 1
        self.t_step %= self.learn_step * self.update_step
        if self.t_step % self.learn_step == 0:
            self.learn()
        if self.t_step % self.update_step == 0:
            self.soft_update()

    def learn(self):
        """
        Update target Q
        """
        states, nxt_states, rewards, dones, acts, idx = \
                                        self.buff.sample(self.batch_size,
                                                         self.device)
        self.qnet_target.eval()
        q_nxt_states = self.qnet_target(nxt_states)\
                                    .detach()\
                                    .max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * q_nxt_states * (1 - dones)
        self.qnet.train()
        acts = acts.long()
        q_exps = self.qnet(states).gather(1, acts)
        loss = F.mse_loss(q_exps, q_targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def soft_update(self, t=1.):
        """
        Soft update target Q to Q
        """
        for param, param_target in zip(self.qnet.parameters(),
                                        self.qnet_target.parameters()):
            param_target.data.copy_(self.t * param.data\
                                    + (1 - self.t) * param_target.data)

    def to(self, device):
        """
        Change device
        """
        self.device = device
        self.q_net = self.q_net.to(device)
        self.q_net_target = self.q_net_target.to(device)

class DoubleDQN_Agent(DQN_Agent):
    def __init__(self, state_size, act_size, config):
        """
        Inherit __init__ from DQN_Agent
        """
        super().__init__(state_size, act_size, config)

    def learn(self):
        """
        Update target Q with Double DQN algo
        """
        states, nxt_states, rewards, dones, acts, idx = \
                                        self.buff.sample(self.batch_size,
                                                         self.device)
        self.qnet.eval()
        est_act = self.qnet(nxt_states).argmax(dim=1)
        self.qnet_target.eval()
        q_nxt_states = self.qnet_target(nxt_states)\
                                    .gather(dim=1, index=est_act.unsqueeze(1))\
                                    .detach()
        q_targets = rewards + self.gamma * q_nxt_states * (1 - dones)
        self.qnet.train()
        acts = acts.long()
        q_exps = self.qnet(states).gather(1, acts)
        loss = F.mse_loss(q_exps, q_targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

class DuelingDQN_Agent(DQN_Agent):
    def __init__(self, state_size, act_size, config):
        super().__init__(state_size, act_size, config)

    def prepare_net(self):
        """
        Change network for Dueling DQN
        """
        self.qnet = DuelingQNet(self.state_size,
                                self.act_size).to(config['device'])
        self.qnet_target = DuelingQNet(self.state_size,
                                       self.act_size).to(config['device'])
        self.opt = torch.optim.AdamW(self.qnet.parameters(),
                                     lr=config['lr'])

class PrioritizeDQN_Agent(DQN_Agent):
    def __init__(self, state_size, act_size, config):
        """
        Change basic random buffer to prioritize replay buffer
        """
        super().__init__(state_size, act_size, config)
        self.buff = PrioritizedBuffer(config['buffer_size'])

    def step(self, state, nxt_state, reward, done, act):
        """
        Update for each step with loss
        """
        # Calculate loss same as Double DQN
        t_state = torch.from_numpy(state).to(self.device)
        t_nxt_state = torch.from_numpy(nxt_state).to(self.device)
        t_reward = torch.Tensor([reward]).to(self.device)
        t_act = torch.Tensor([act]).to(self.device)
        self.qnet.eval()
        est_act = self.qnet(t_nxt_state).argmax()
        self.qnet_target.eval()
        q_nxt_state = self.qnet_target(t_nxt_state)\
                                  .gather(dim=0, index=est_act)
        q_target = t_reward + self.gamma * q_nxt_state * (1 - done)
        self.qnet.train()
        t_act = t_act.long()
        q_exp = self.qnet(t_state).gather(dim=0, index=t_act)
        loss = F.mse_loss(q_exp, q_target).item()

        self.buff.store(state, nxt_state, reward, done, act, loss)
        self.t_step += 1
        self.t_step %= self.learn_step * self.update_step
        if self.t_step % self.learn_step == 0:
            self.learn()
        if self.t_step % self.update_step == 0:
            self.soft_update()

    def learn(self):
        """
        Update target Q with Double DQN algo
        """
        states, nxt_states, rewards, dones, acts, idx = \
                                        self.buff.sample(self.batch_size,
                                                         self.device)
        self.qnet.eval()
        est_act = self.qnet(nxt_states).argmax(dim=1)
        self.qnet_target.eval()
        q_nxt_states = self.qnet_target(nxt_states)\
                                    .gather(dim=1, index=est_act.unsqueeze(1))\
                                    .detach()
        q_targets = rewards + self.gamma * q_nxt_states * (1 - dones)
        self.qnet.train()
        acts = acts.long()
        q_exps = self.qnet(states).gather(1, acts)
        loss = F.mse_loss(q_exps, q_targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.buff.update_loss(idx, (q_exps - q_targets).pow(2).sum(dim=1) / q_exps.size(1))

# TODO: Test NoisyNetAgent
class NoisyNetAgent(DQN_Agent):
    def __init__(self, state_size, act_size, config):
        super().__init__(state_size, act_size, config)

    def prepare_net(self):
        """
        change QNet to NouisyNet
        """
        self.qnet = NoisyNet(self.state_size,
                         self.act_size).to(config['device'])
        self.qnet_target = NoisyNet(self.state_size,
                                self.act_size).to(config['device'])
        self.opt = torch.optim.AdamW(self.qnet.parameters(),
                                     lr=config['lr'])

    def act(self, state, eps=0.):
        state = torch.from_numpy(state)\
                    .float()\
                    .unsqueeze(0)\
                    .to(self.device)
        self.qnet.eval()
        with torch.no_grad():
            act_vals = self.qnet(state)
        return np.argmax(act_vals.cpu().data.numpy())

    def step(self, state, nxt_state, reward, done, act):
        """
        Update for each step
        """
        self.buff.store(state, nxt_state, reward, done, act)
        self.t_step += 1
        self.t_step %= self.learn_step * self.update_step
        if self.t_step % self.learn_step == 0:
            self.learn()
        if self.t_step % self.update_step == 0:
            self.soft_update()

    def learn(self):
        """
        Update target Q
        """
        states, nxt_states, rewards, dones, acts, idx = \
                                        self.buff.sample(self.batch_size,
                                                         self.device)
        self.qnet_target.set_noise()
        self.qnet_target.eval()
        q_nxt_states = self.qnet_target(nxt_states)\
                                    .detach()\
                                    .max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * q_nxt_states * (1 - dones)
        self.qnet.set_noise()
        self.qnet.train()
        acts = acts.long()
        q_exps = self.qnet(states).gather(1, acts)
        loss = F.mse_loss(q_exps, q_targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
