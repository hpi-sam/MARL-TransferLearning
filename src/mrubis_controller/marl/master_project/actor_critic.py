import torch
import torch.nn as nn


class A2CNet(nn.Module):
    def __init__(self, n_actions, actor_lr, critic_lr, layer_dims=None):
        super().__init__()
        self.n_actions = n_actions
        self.layer_dims = [36, 72] if layer_dims is None else layer_dims
        self.build_network()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value

    def build_network(self):
        layers = []
        for index, dims in enumerate(self.layer_dims):
            if index == 0:
                layers.append(torch.nn.Linear(self.n_actions, dims))
            else:
                layers.append(torch.nn.Linear(
                    self.layer_dims[index - 1], dims))
            layers.append(torch.nn.ReLU())
        actor_layers = layers.copy()
        critic_layers = layers.copy()
        actor_layers.append(torch.nn.Linear(
            self.layer_dims[-1], self.n_actions))
        actor_layers.append(torch.nn.Softmax(dim=1))  # ? Ist dim=1 korrekt?
        critic_layers.append(torch.nn.Linear(self.layer_dims[-1], 1))

        self.critic = torch.nn.Sequential(*critic_layers)
        self.actor = torch.nn.Sequential(*actor_layers)