import torch
import torch.nn as nn

class ActorCritic():
    def __init__(self, state_dim, act_dim, emb_size=64, continuous_space, act_std):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.continuous_space = continuous_space
        self.act_std = act_std
        sefl.set_act_var()
        
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, emb_size),
                        nn.ReLU(),
                        nn.Linear(emb_size, emb_size),
                        nn.ReLU(),
                        nn.Linear(emb_size, act_dim),
                     )

        self.critic = nn.Sequential(
                        nn.Linear(state_dim, emb_size),
                        nn.ReLU(),
                        nn.Linear(emb_size, emb_size),
                        nn.ReLU(),
                        nn.Linear(emb_size, 1),
                     )

    def set_act_var(self):
        """
        Update self.act_var with self.act_std
        """
        self.act_var = torch.full((self.act_dim,), act_std * act_std)

    def gen_dist(self, state):
        """
        Generate distribution of actions for the state
        input: state
        output: dist
        """
        if self.continuous_space:
            act_mean = self.actor(state)
            act_var = self.act_var.expand_as(act_mean).to(self.device)
            cov_mtx = torch.diag_embed(act_var).to(self.device)
            dist = torch.distrubution.MultivariateNormal(act_mean, cov_mtx)
        else:
            act_prob = self.actor(state)
            dist = torch.distrubution.Categorical(act_prob)
        return dist

    def act(self, state, detach=True):
        """
        Action take by self.actor
        input: state
        output: act, act_logprob
        """
        dist = self.gen_dist(state)
        act = dist.sample()
        act_logprob = dist.log_prob(act)
        return act.detach(), act_logprob.detach()

    def eval(self, state, act):
        """
        Evaluate old actions and values
        """
        dist = self.gen_dist(state)
        act_logprob = dist.log_prob(act)
        dist_entropy = dist.entropy
        state_val = self.critic(state)
        return act_logprob, state_val, dist_entropy

    def sync_param(self, target):
        """
        Syncronize models' parameters to target's
        """
        self.actor.load_state_dict(target.actor.state_dict())
        self.critic.load_state_dict(target.critic.state_dict())

    def to(self, device):
        """
        Change device
        """
        self.device = device
        self.agent = self.agent.to(device)
        self.actor = self.actor.to(device)

#TODO: implement PPO
class PPO():
    def __init__(self, state_size, act_size, config):
        self.state_size = state_size
        self.act_size = act_size
