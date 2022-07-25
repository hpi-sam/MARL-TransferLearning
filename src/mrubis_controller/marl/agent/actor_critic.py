from copy import deepcopy
from typing import Tuple
import numpy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from marl.agent.actor import CombinedActor, LinearActor
from marl.agent.critic import CombinedCritic, LinearCritic


class ActorCritic:
    def __init__(self):
        base_model = [
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 72),
            torch.nn.ReLU(),
        ]
        self.actor = LinearActor()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 1e-3)
        self.c1 = LinearCritic()
        self.c2 = LinearCritic()
        self.c1_optimizer = torch.optim.Adam(self.c1.parameters(), 1e-3)
        self.c2_optimizer = torch.optim.Adam(self.c2.parameters(), 1e-3)

        self.target_c1 = LinearCritic()
        self.target_c2 = LinearCritic()

        self.target_entropy = -np.log((1.0 / 18)) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)

        self.alpha = 0.2
        self.discount_rate = 0.99
        self.tau = 0.995

    def get_action(self, state: torch.Tensor, with_shield=True, allowed_actions=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve an action from an observation.

        Args:
            state (torch.Tensor): The encoded observation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns the sampled action, the probabilites and the log probabilities.
        """
        return self.actor(state, with_shield = with_shield, allowed_actions=allowed_actions)
    
    def learn(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor
    ):
        self.learn_q(observation, action, reward, next_observation)
        self.learn_pi(observation)
    
    def learn_q(self, observation: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_observation: torch.Tensor):
        with torch.no_grad():
            next_action, next_action_p, next_action_log_p = self.actor(next_observation)
            q1_target = self.target_c1(next_observation)
            q2_target = self.target_c2(next_observation)
            y = next_action_p * (torch.min(q1_target, q2_target) - self.alpha * next_action_log_p)
            y = y.sum(dim=1).unsqueeze(-1)
            next_q_value = reward.reshape(y.shape) +  self.discount_rate * (y)

        self.c1_optimizer.zero_grad()
        self.c2_optimizer.zero_grad()

        q1 = self.c1(observation).gather(1, action.long())
        q2 = self.c2(observation).gather(1, action.long())
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)

        q1_loss.backward()
        q2_loss.backward()

        self.c1_optimizer.step()
        self.c2_optimizer.step()
        
        self.update_target(self.target_c1, self.c1)
        self.update_target(self.target_c2, self.c2)

    def learn_pi(self, observation: torch.Tensor):
        next_action, next_action_p, next_action_log_p = self.actor(observation)
        q1 = self.c1(observation)
        q2 = self.c2(observation)
        min_q = torch.min(q1, q2)
        inner_term = self.alpha * next_action_log_p - min_q
        pi_loss = (next_action_p * inner_term).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        pi_loss.backward()

    def update_target(self, target: torch.nn.Module, follower: torch.nn.Module):
        for target_param, local_param in zip(target.parameters(), follower.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

class SimpleActorCritic:
    # No target network
    def __init__(self):
        base_model = [
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 72),
            torch.nn.ReLU(),
        ]
        self.actor = CombinedActor(base_model, 72)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = CombinedCritic(base_model, 72)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.target_critic = CombinedCritic(deepcopy(base_model), 72)

        self.learn_alpha = True
        if not self.learn_alpha:
            self.alpha = 0.2
        else:
            self.target_entropy = -numpy.log((1.0 / 18)) * 0.98
            self.log_alpha = torch.nn.parameter.Parameter(torch.zeros(1, requires_grad=True))
            self.alpha = torch.nn.parameter.Parameter(self.log_alpha.exp())
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.discount_rate = 0.99
        self.tau = 0.995

    def get_action(self, state: torch.Tensor, with_shield=True, allowed_actions=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve an action from an observation.

        Args:
            state (torch.Tensor): The encoded observation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns the sampled action, the probabilites and the log probabilities.
        """
        return self.actor(state, with_shield = with_shield, allowed_actions=allowed_actions)
    
    def learn(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor
    ):
        self.learn_q(observation, action, reward, next_observation)
        self.learn_pi(observation)
    
    def learn_q(self, observation: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_observation: torch.Tensor):
        with torch.no_grad():
            next_action, next_action_p, next_action_log_p = self.actor(next_observation)
            q_target = self.critic(next_observation)
            y = next_action_p * (q_target - self.alpha * next_action_log_p)
            y = y.sum(dim=1).unsqueeze(-1)
            next_q_value = reward.reshape(y.shape) +  self.discount_rate * (y)

        self.critic_optimizer.zero_grad()

        q = self.critic(observation).gather(1, action.long())
        q_loss = F.mse_loss(q, next_q_value)

        q_loss.backward()

        self.critic_optimizer.step()
        self.update_target(self.target_critic, self.critic)

    def learn_pi(self, observation: torch.Tensor):
        next_action, next_action_p, next_action_log_p = self.actor(observation)
        q = self.critic(observation)
        inner_term = self.alpha * next_action_log_p - q
        pi_loss: torch.Tensor = (next_action_p * inner_term).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()

        if self.learn_alpha:
            self.alpha_optim.zero_grad()
            alpha_loss = -(self.log_alpha * (next_action_log_p + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

    def update_target(self, target: torch.nn.Module, follower: torch.nn.Module):
        for target_param, local_param in zip(target.parameters(), follower.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
