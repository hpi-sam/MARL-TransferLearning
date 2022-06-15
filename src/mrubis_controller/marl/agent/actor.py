from torch.nn import Module
import torch.nn as nn
import torch

class Actor(Module):
    def __init__(self, observation_length=18, action_length=18) -> None:
        super().__init__()
    
    def clone(self):
        raise NotImplementedError()

class LSTMActor(Module):
    def __init__(self, input_size, hidden_size, num_layers, num_actions):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, inputs, hidden):
        """
        inputs: (batch_size, seq_len, input_size)
        hidden: (num_layers, batch_size, hidden_size)
        """
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def select_actions(self, observations: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        observations: (batch_size, seq_len, input_size)
        """
        output, hidden = self.forward(observations, hidden)
        return output, hidden
    
    def clone(self):
        return LSTMActor(self.input_size, self.hidden_size, self.num_layers, self.num_actions)

class LinearActor(Actor):
    def __init__(self, observation_length=18, action_length=18, activation=torch.nn.Tanh, dimensions=[18,18,18]):
        super(LinearActor, self).__init__(observation_length=18, action_length=18)
        self.observation_length=observation_length
        self.action_length=action_length
        self.activation=activation
        self.dimensions=[18,18,18]
        layers = []
        for i in range(len(dimensions)):
            if i == 0:
                layers.append(nn.Linear(observation_length, dimensions[i]))
            else:
                layers.append(nn.Linear(dimensions[i-1], dimensions[i]))
            if activation is not None:
                layers.append(activation())
            layers.append(nn.Linear(dimensions[-1], action_length))
        self.model = nn.Sequential(*layers, torch.nn.Sigmoid())

    def forward(self, observations: torch.Tensor):
        """
        observations: (batch_size, observation_length)
        """
        return self.model(observations)

    def clone(self):
        return LinearActor(self.observation_length, self.action_length, self.activation, self.dimensions)