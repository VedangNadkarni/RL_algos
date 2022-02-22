import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.functional as F


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Net(nn.Module):

    def __init__(self, c, h, w, outputs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        self.convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        self.convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.outputs = outputs
        linear_input_size = self.convw * self.convh * 32
        if self.convw * self.convh > 3 * outputs:
            self.head_ = nn.Linear(linear_input_size, self.convw * self.convh)
            self.head = nn.Linear(self.convw * self.convh, outputs)
        else:
            self.head_ = nn.Linear(linear_input_size, outputs)
        # self.out = F.softmax

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x)
        # x = x.to(device)
        # print('U', x[0])
        # print('O', np.shape(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        if self.convw * self.convh > 3 * self.outputsoutputs:
            return self.head_(F.ReLu(self.head_(x.view(x.size(0), -1))))
        else:
            return self.head_(x.view(x.size(0), -1))

class Actor(nn.Module):
    
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class CategoricalActor(Actor):
    def __init__(self, c, h, w, act_dim = 1):
        super().__init__()
        self.logits_net = Net(c, h, w, act_dim)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        # print(logits)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class GaussianActor(Actor):

    def __init__(self, h, w, act_dim = 1):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = Net(h, w, act_dim)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class Critic(nn.Module):

    def __init__(self, c, h, w):
        super().__init__()
        self.v_net = Net(c, h, w, 1)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class ActorCritic(nn.Module):


    def __init__(self, c, h, w, action_space):
        super().__init__()

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = GaussianActor(c, h, w, action_space.n)
        elif isinstance(action_space, Discrete):
            self.pi = CategoricalActor(c, h, w, action_space.n)

        # build value function
        self.v  = Critic(c, h, w)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]