from typing import Tuple
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from marl.agent.actor import LinearActor
from marl.agent.critic import LinearCritic


class ActorCritic:
    def __init__(self):
        self.actor = LinearActor()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 1e-3)
        self.c1 = LinearCritic()
        self.c2 = LinearCritic()
        self.c1_optimizer = torch.optim.Adam(self.c1.parameters(), 1e-3)
        self.c2_optimizer = torch.optim.Adam(self.c2.parameters(), 1e-3)

        self.target_c1 = LinearCritic()
        self.target_c2 = LinearCritic()

        self.alpha = 0.8
        self.discount_rate = 0.99
        self.tau = 0.9999

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
