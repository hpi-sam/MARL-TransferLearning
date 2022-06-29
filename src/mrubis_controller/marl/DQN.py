from torch.nn import Module
import torch.nn as nn
import torch


class LinearNeuralNetwork(Module):
    def __init__(self, observation_length=270, action_length=18, activation=torch.nn.Tanh, dimensions=[50]):
        super().__init__()
        self.observation_length = observation_length
        self.action_length = action_length
        self.activation = activation
        self.dimensions = dimensions
        layers = []
        for i in range(len(dimensions)):
            if i == 0:
                layers.append(nn.Linear(observation_length, dimensions[i]))
            else:
                layers.append(nn.Linear(dimensions[i-1], dimensions[i]))
            if activation is not None:
                layers.append(activation())
        layers.append(nn.Linear(
            dimensions[-1] if len(dimensions) > 0 else observation_length, action_length))
        self.model = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor):
        """
        observations: (batch_size, observation_length)
        """
        observations -= observations.min()
        observations /= observations.max()
        output = self.model(observations)
        # return output
        return torch.softmax(output, dim=1 if len(output.shape) > 1 else 0)

    def clone(self):
        return LinearNeuralNetwork(self.observation_length, self.action_length, self.activation, self.dimensions)
