# follows https://dev.to/jemaloqiu/reinforcement-learning-with-tf2-and-gym-actor-critic-3go5
import numpy
import numpy as np
import torch

from marl.master_project.helper import get_current_time
from marl.master_project.sorting.agent_action_sorter import AgentActionSorter
from entities.component_failure import ComponentFailure
from entities.observation import ComponentObservation, ShopObservation, SystemObservation
from marl.master_project.actor_critic import A2CNet
from marl.replay_buffer import ReplayBuffer


def _decoded_action(action, observation):
    return list([components.components.keys() for shop, components in observation.items()][0])[action]


def encode_observations(observations):
    encoded_observations = [
        1 if component.failure_name != ComponentFailure.NONE else 0
        for component in observations.components.values()
    ]

    return np.array(encoded_observations, dtype=float)


def get_root_cause(observations):
    observations = observations.components
    for idx, component in enumerate(observations.values()):
        if component.root_issue is True:
            return idx, list(observations.keys())[idx]


def output_probabilities(probabilities):
    output = []
    for index, p in enumerate(probabilities):
        if p >= 0.01:
            output.append('\033[92m' + str(index) + ": " +
                          "{:.2f}".format(p) + '\033[0m')
        else:
            output.append(str(index) + ": " + "{:.2f}".format(p))
    # print(' '.join(output))

class Agent:
    def __init__(self, shops, action_space_inverted, load_models_data, ridge_regression_train_data_path, index=0,
                 lr=0.001, layer_dims=None, training_activated=True):
        self.index = index
        self.shops = shops
        self.train = training_activated
        self.base_model_dir = './mrubis_controller/marl/data/models'
        self.base_log_dir = './mrubis_controller/marl/data/logs/'
        self.start_time = get_current_time()

        self.load_models_data = load_models_data
        self.ridge_regression_train_data_path = ridge_regression_train_data_path

        self.action_space_inverted = list(action_space_inverted)
        self.gamma = 0.99
        self.alpha = lr
        self.beta = 0.0005
        self.n_actions = len(action_space_inverted)
        self.input_dims = self.n_actions
        self.layer_dims = [36, 72] if layer_dims is None else layer_dims

        self.replay_buffers = {shop: ReplayBuffer(100) for shop in self.shops}

        self.model = A2CNet(self.n_actions, self.alpha,
                            self.beta, self.layer_dims)

        self.action_space = list(range(self.n_actions))

        # stage 0 = no sorting as a baseline
        # stage 1 = sorting of actions
        self.stage = 1

        self.agent_action_sorter = AgentActionSorter(
            self.ridge_regression_train_data_path)
        self.memory = {}
        self.previous_actions = {}

    def choose_action(self, observations):
        """ chooses actions based on observations
            each trace will be introduced to the network to find a fix
            sort actions afterwards for maximizing utility
            returns sorted list
        """
        actions = []
        regrets = {}
        root_causes = {}
        for shop_name, components in observations.items():
            regret = None
            state = encode_observations(components)[np.newaxis, :]
            state_tensor = torch.from_numpy(state).float()
            if state.sum() > 0:  # skip shops without any failure
                probabilities, _ = self.model(state_tensor)
                probabilities = probabilities.detach().numpy()[0]
                action = numpy.argmax(probabilities)
                probability = probabilities[action]
                root_cause_index, root_cause_name = get_root_cause(components)
                regret = 1.0 - probabilities[root_cause_index]
                decoded_action = _decoded_action(action, observations)
                step = {'shop': shop_name, 'component': decoded_action}
                if self.stage >= 1:
                    step['predicted_utility'] = self.agent_action_sorter.predict_optimal_utility_of_fixed_components(
                        step, components)
                if self.stage == 2:
                    # reduce predicted utility by uncertainty
                    step['predicted_utility'] *= probability
                actions.append(step)
                regrets[shop_name] = regret
                root_causes[shop_name] = root_cause_name
        return actions, regrets, root_causes

    def choose_from_memory(self, state, shop_name, components):
        if self.obs_in_memory(shop_name, components):
            action = numpy.argmax(self.previous_actions[shop_name])
            probability = self.previous_actions[shop_name][action]
            self.previous_actions[shop_name][action] = -1
        else:
            state_tensor = torch.from_numpy(state).float()
            probabilities, _ = self.model(state_tensor)
            probabilities = probabilities.detach().numpy()[0]
            action = numpy.argmax(probabilities)
            probability = probabilities[action]
            probabilities[action] = 0
            self.previous_actions[shop_name] = probabilities
        return action, probability

    def learn(self, batch_size: int = 1):
        """ network learns to improve """
        metrics = []
        for shop_name in self.shops:
            observations, actions, selected_actions, rewards, next_observations  = self.replay_buffers[shop_name].get_batch(batch_size)
            if observations is None:
                continue
            torch.autograd.set_detect_anomaly(True)
            y_pred, critic_value = self.model(observations)
            critic_value = critic_value.detach().numpy()
            delta = rewards - critic_value

            loss = torch.nn.MSELoss()
            # ToDo check whether rewards is correctly used here, or should be a new tensor
            output: torch.Tensor = loss(torch.tensor(critic_value, requires_grad=True).to(torch.float32),
                          rewards.requires_grad_())
            self.model.critic_optimizer.zero_grad()

            out = torch.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = torch.log(out).gather(1, torch.unsqueeze(selected_actions, 1))
            # delta = torch.tensor(delta)
            # ToDo: Check if this makes sense, or whether we should remove the inner sum
            actor_loss = torch.mean(torch.sum(-log_lik * delta, dim=0), dim=0)

            self.model.actor_optimizer.zero_grad()
            output.backward()
            actor_loss.backward()
            self.model.critic_optimizer.step()
            self.model.actor_optimizer.step()

            metrics.append({"actor": float(actor_loss),
                           "critic": float(output)})
        return metrics

    def get_probabilities_for_observations(self, observations):
        return self.policy.predict(encode_observations(observations)[np.newaxis, :])[0]

    def obs_in_memory(self, shop_name, components: ShopObservation):
        failing_components = {component: components.components[component].shop_utility
                              for component in components.components.keys() if components.components[component].failure_name is not ComponentFailure.NONE}
        same_obs = False
        if shop_name not in self.memory:
            same_obs = False
        elif all(component in self.memory[shop_name] and utility < self.memory[shop_name][component]
                 for component, utility in failing_components.items()):
            same_obs = True
        self.memory[shop_name] = failing_components
        return same_obs

    def save(self, episode):
        self.actor.save(
            f"{self.base_model_dir}/{self.start_time}/agent_{self.index}/actor/episode_{episode}")
        self.critic.save(
            f"{self.base_model_dir}/{self.start_time}/agent_{self.index}/critic/episode_{episode}")

    def remove_shops(self, shops):
        self.shops = self.shops.difference(shops)

    def add_shops(self, shops):
        self.shops = self.shops.union(shops)

    def add_to_replay_buffer(self, states, actions, reward, next_states, dones):
        for action_idx, action in actions.items():
            shop_name = action["shop"]
            if shop_name in self.shops:
                #ToDo: Masterprojekt only using action[component] here !
                # ToDo: check if we are really using the action_index correctly (before was argmax(action))
                state = encode_observations(states[shop_name])
                state_tensor = torch.from_numpy(state).float()
                next_state = encode_observations(next_states[shop_name])
                next_state_tensor = torch.from_numpy(next_state).float()
                reward_tensor = torch.unsqueeze(torch.tensor(reward[0][shop_name]), 0)
                # ToDo: Insert actual actions, but we need them encoded to use them as tensor
                self.replay_buffers[shop_name].add(state_tensor, torch.tensor(0), torch.tensor(action_idx), reward_tensor, next_state_tensor)