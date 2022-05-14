import numpy as np
from mrubis_controller.marl.sorting.agent_action_sorter import AgentActionSorter


def _decoded_action(action, observation):
    return list([components.keys() for shop, components in observation.items()][0])[action]


def encode_observations(observations):
    encoded_observations = [
        1 if component['failure_name'] != 'None' else 0
        for component in observations.values()
    ]

    return np.array(encoded_observations, dtype=float)


class RandomAgent:
    def __init__(self, shops, action_space_inverted, load_models_data, ridge_regression_train_data_path, index=0,
                 lr=0.003, layer_dims=None):
        self.index = index
        self.shops = shops

        self.ridge_regression_train_data_path = ridge_regression_train_data_path

        self.action_space_inverted = list(action_space_inverted)
        self.n_actions = len(action_space_inverted)
        self.action_space = list(range(self.n_actions))

        # stage 0 = no sorting as a baseline
        # stage 1 = sorting of actions
        self.stage = 1

        self.agent_action_sorter = AgentActionSorter(self.ridge_regression_train_data_path)

    def choose_action(self, observations):
        """
        chooses random action from failing components
        """
        actions = []
        for shop_name, components in observations.items():
            state = encode_observations(components)[np.newaxis, :]
            if state.sum() > 0:  # skip shops without any failure
                probabilities = state[0] / sum(state[0])
                action = np.random.choice(self.action_space, p=probabilities)
                decoded_action = _decoded_action(action, observations)
                step = {'shop': shop_name, 'component': decoded_action}
                if self.stage >= 1:
                    step['predicted_utility'] = self.agent_action_sorter.predict_optimal_utility_of_fixed_components(
                        step, components)
                if self.stage == 2:
                    # reduce predicted utility by uncertainty
                    step['predicted_utility'] *= probabilities[action]
                actions.append(step)
        return actions

    def learn(self, states, actions, reward, states_, dones):
        pass

    def get_probabilities_for_observations(self, observations):
        state = encode_observations(observations)[np.newaxis, :]
        probabilities = state[0] / sum(state[0])
        return probabilities

    def save(self, episode):
        pass

    def remove_shops(self, shops):
        self.shops = self.shops.difference(shops)

    def add_shops(self, shops):
        self.shops = self.shops.union(shops)
