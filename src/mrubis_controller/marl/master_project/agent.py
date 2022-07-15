# follows https://dev.to/jemaloqiu/reinforcement-learning-with-tf2-and-gym-actor-critic-3go5
import numpy
import numpy as np
import torch

from marl.master_project.helper import get_current_time
from marl.master_project.sorting.agent_action_sorter import AgentActionSorter
from entities.component_failure import ComponentFailure
from entities.observation import ComponentObservation, ShopObservation, SystemObservation
from marl.master_project.actor_critic import A2CNet


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
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)

        # self.actor, self.critic, self.policy = self._build_network()
        self.model = A2CNet(self.n_actions, self.alpha,
                            self.beta, self.layer_dims)
        """
        self.optimizer = torch.optim.Adam(  # !Delete perspektivisch
            self.actor.parameters(), lr=self.alpha)  # use parameters from actor or critic??? sollte eher actor sein
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.alpha)  # use parameters from actor or critic??? sollte eher actor sein
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.beta)  # use parameters from actor or critic??? sollte eher actor sein
        """
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
                # probabilities = self.policy.predict(state)[0]
                probabilities, _ = self.model(state_tensor)
                probabilities = probabilities.detach().numpy()[0]
                # if self.train:
                action, probability = self.choose_from_memory(
                    state, shop_name, components)
                # else:
                #     action = np.random.choice(self.action_space, p=probabilities)
                #     probability = probabilities[action]
                output_probabilities(probabilities)
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
            # probabilities = self.policy.predict(state)[0]
            state_tensor = torch.from_numpy(state).float()
            probabilities, _ = self.model(state_tensor)
            probabilities = probabilities.detach().numpy()[0]
            action = numpy.argmax(probabilities)
            probability = probabilities[action]
            probabilities[action] = 0
            self.previous_actions[shop_name] = probabilities

        return action, probability

    def learn(self, states, actions, reward, states_, dones):
        """ network learns to improve """
        actions = {action['shop']: action['component']
                   for action in actions.values()}
        metrics = []
        for shop_name, action in actions.items():
            torch.autograd.set_detect_anomaly(True)
            state = encode_observations(states[shop_name])[np.newaxis, :]
            # state_ = encode_observations(states_[shop_name])[np.newaxis, :]

            # critic_value = self.critic.predict(state)  # !Das ist tf
            state_tensor = torch.from_numpy(state).float()

            y_pred, critic_value = self.model(state_tensor)  # Ist das richtig?
            critic_value = critic_value.detach().numpy()
            #critic_value[0][0] = double(critic_value[0][0])
            #y_pred = y_pred.detach().numpy()[0]
            # critic_value_ = self.critic.predict(state_)

            # TODO: How important is the length of an episode? Is there a future reward of a solved state?
            shop_reward = reward[0][shop_name]
            target = np.reshape(shop_reward, (1, 1))

            delta = target - critic_value

            # TODO reset in chosen_action
            if shop_reward > 0:
                del self.memory[shop_name]

            _actions = np.zeros([1, self.n_actions])
            _actions[np.arange(
                1), self.action_space_inverted.index(action)] = 1.0

            # , callbacks=[self.tb_callback])
            # critic_history = self.critic.fit(state, target, verbose=0)
            # calculating loss for the critic
            loss = torch.nn.MSELoss()
            #output = loss(state, target)
            output = loss(torch.tensor(critic_value, requires_grad=True).to(torch.float64),
                          torch.tensor(target, requires_grad=True))
            self.model.critic_optimizer.zero_grad()

            # calculating loss for the actor
            # y_pred = self.model.actor([state, delta])
            out = torch.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = torch.tensor(_actions) * torch.log(out)
            delta = torch.tensor(delta)
            actor_loss = torch.sum(-log_lik * delta)
            # loss = compute_loss(model, x)
            self.model.actor_optimizer.zero_grad()
            output.backward()
            actor_loss.backward()
            self.model.critic_optimizer.step()
            self.model.actor_optimizer.step()
            # das ist das alte
            # grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            # if self.train:
            # self.optimizer.apply_gradients(
            #    zip(grads, self.actor.trainable_variables))

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

    """
    def _build_network(self):
        if self.load_models_data is not None:
            return self.load_models(self.load_models_data)
        model_input = torch.nn.Linear(self.input_dims,)  # name='input')
        delta = Input(shape=[1], name='delta')

        dense_layer = model_input
        layers = []
        for index, dims in enumerate(self.layer_dims):
            if index == 0:
                layers.append(torch.nn.Linear(self.input_dims, dims))
            else:
                layers.append(torch.nn.Linear(
                    self.layer_dims[index - 1], dims))
            layers.append(torch.nn.ReLU())
        actor_layers = layers
        actor_layers.append(torch.nn.Linear(
            self.layer_dims[-1], self.n_actions))
        actor_layers.append(torch.nn.Softmax())
        critic_layers = layers
        critic_layers.append(torch.nn.Linear(self.layer_dims[-1], 1))
        critic_layers.append(torch.nn.Linear())
        # values = Dense(1, activation='linear', name='values')(dense_layer)

        # actor = Model(inputs=[model_input, delta], outputs=[probs])
        actor = torch.nn.Sequential(*actor_layers)
        # critic = Model(inputs=[model_input], outputs=[values])
        critic = torch.nn.Sequential(*critic_layers)
        # wie geht das mit dem loss?
        # critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')#! Das ist der critic mit loss und lr

        policy = Model(inputs=[model_input], outputs=[probs])
        print(actor.summary())
        print(critic.summary())
        print(policy.summary())
        return actor, critic, policy
    """

    def save(self, episode):
        self.actor.save(
            f"{self.base_model_dir}/{self.start_time}/agent_{self.index}/actor/episode_{episode}")
        self.critic.save(
            f"{self.base_model_dir}/{self.start_time}/agent_{self.index}/critic/episode_{episode}")
    """
    def load_models(self, load_models_data):
        base_dir = f"{self.base_model_dir}/{load_models_data['start_time']}/agent_{self.index}"

        # load critic
        critic = tf.keras.models.load_model(
            f"{base_dir}/critic/episode_{load_models_data['episode']}")

        # load actor and set layers
        actor_copy = tf.keras.models.load_model(
            f"{base_dir}/actor/episode_{load_models_data['episode']}")
        probs = actor_copy.get_layer('probs')(
            critic.get_layer('layer_last').output)
        actor = Model(inputs=[critic.get_layer('input').input,
                      actor_copy.get_layer('delta').input], outputs=[probs])

        # load policy
        policy = Model(inputs=[critic.get_layer(
            'input').input], outputs=[probs])

        return actor, critic, policy
    """

    def remove_shops(self, shops):
        self.shops = self.shops.difference(shops)

    def add_shops(self, shops):
        self.shops = self.shops.union(shops)
