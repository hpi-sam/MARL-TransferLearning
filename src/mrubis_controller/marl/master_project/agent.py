# follows https://dev.to/jemaloqiu/reinforcement-learning-with-tf2-and-gym-actor-critic-3go5
from typing import Dict
import logging
import numpy
import numpy as np
import torch
import torch.nn.functional as F
from entities.components import Components
from marl.options import args

from marl.master_project.helper import get_current_time
from marl.master_project.sorting.agent_action_sorter import AgentActionSorter
from entities.component_failure import ComponentFailure
from entities.observation import ShopObservation, SystemObservation
from marl.master_project.actor_critic import A2CNet
from marl.replay_buffer import ReplayBuffer
from marl.options import args


def _decoded_action(action, observation: SystemObservation):
    return Components.value_list()[action]


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
        self.sampled_actions: Dict[str, 'list[int]'] = {}

    def sample_random_action(self, shop_name, probabilities):
        p = probabilities.copy()
        p_new = p
        if shop_name in self.sampled_actions.keys():
            total = np.float64(0)
            for action in self.sampled_actions[shop_name]:
                total += p[action]
                p[action] = 0
            c =  len(p) - len(self.sampled_actions[shop_name])
            indices = [i for i in range(0, len(p)) if i not in self.sampled_actions[shop_name]]
            np.add.at(p_new, indices, total/c)
        
        return np.random.choice([i for i in range(0, len(p_new))], 1, p=p_new)

    def choose_action(self, observations):
        """ chooses actions based on observations
            each trace will be introduced to the network to find a fix
            sort actions afterwards for maximizing utility
            returns sorted list
        """
        actions = []
        regrets = {}
        root_causes = {}
        action_probabilities = {}
        for shop_name, components in observations.items():
            if shop_name not in self.shops:
                continue
            regret = None
            state = encode_observations(components)[np.newaxis, :]
            state_tensor = torch.from_numpy(state).float()
            if state.sum() > 0:  # skip shops without any failure
                probabilities, _ = self.model(state_tensor)
                probabilities = probabilities.detach().numpy()[0]
                action_probabilities[shop_name] = probabilities

                if args.use_exploration:
                    action = self.sample_random_action(shop_name, probabilities).item()
                    if shop_name in self.sampled_actions.keys():
                        self.sampled_actions[shop_name].append(action)
                    else:
                        self.sampled_actions[shop_name] = [action]
                else:
                    self.choose_from_memory(
                        state, shop_name, components)
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
        logging.debug(action_probabilities)
        return actions, regrets, root_causes, action_probabilities

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
    
    def on_policy_loss(self, state: torch.Tensor, selected_action: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        y_pred, critic_value = self.model(state)
        critic_value = critic_value.detach()
        critic_value.requires_grad = True
        delta = reward - critic_value
        _actions = torch.zeros([1, self.n_actions])
        _actions[:, selected_action] = 1.0
        critic_loss = F.mse_loss(critic_value,reward)
        out = torch.clip(y_pred, 1e-8, 1 - 1e-8)
        log_likelihood = _actions * torch.log(out)
        actor_loss = torch.sum(-log_likelihood * delta)
        return actor_loss, critic_loss

    def learn_on_policy(self, states, actions, reward, states_, dones):
        """ network learns to improve """
        actions = {
            action['shop']: action['component']
                   for action in actions.values()
            }
        metrics = []
        for shop_name, action in actions.items():
            torch.autograd.set_detect_anomaly(True)
            state = encode_observations(states[shop_name])[np.newaxis, :]
            state_tensor = torch.from_numpy(state).float()
            y_pred, critic_value = self.model(state_tensor)  # Ist das richtig?
            critic_value = critic_value.detach().numpy()

            # TODO: How important is the length of an episode? Is there a future reward of a solved state?
            shop_reward = reward[0][shop_name]
            target = np.reshape(shop_reward, (1, 1))

            delta = target - critic_value

            # TODO reset in chosen_action
            if shop_reward > 0:
                del self.memory[shop_name]

            _actions = np.zeros([1, self.n_actions])
            _actions[np.arange(1), self.action_space_inverted.index(action)] = 1.0
            loss = torch.nn.MSELoss()
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

            metrics.append({"actor": float(actor_loss),
                           "critic": float(output)})
        return metrics

    def off_policy_actor_loss(self, observation, action, selected_action, rewards, next_observation):
        p_new, v_new = self.model(observation)
        p_old = action.gather(1, selected_action.unsqueeze(-1))
        p_old_new = p_new.gather(1, selected_action.unsqueeze(-1))
        value_o_t = v_new.detach()
        _, value_o_t1 = self.model(next_observation)
        value_o_t1.detach()

        loss = torch.sum(-(p_old_new / p_old) * torch.log(p_old_new)*(rewards+0.99 * value_o_t1 - value_o_t))
        return loss

    def off_policy_critic_loss(self, observation, action, selected_action, rewards, next_observation):
        p_new, v_new = self.model(observation)
        p_old = action.gather(1, selected_action.unsqueeze(-1)).detach()
        p_old_new = p_new.gather(1, selected_action.unsqueeze(-1)).detach()
        value_o_t = v_new
        _, value_o_t1 = self.model(next_observation)
        y = rewards + 0.99 * value_o_t1
        loss = torch.sum(-(p_old_new / p_old) * torch.pow(torch.sum(torch.abs(value_o_t - y)), 2))
        return loss

    def off_policy_losses_policy_gradient(self, batch_observation, batch_action, batch_selected_action, batch_rewards, batch_next_observation):
        y_pred_batch, _ = self.model(batch_observation)
        y_pred_batch = torch.clip(y_pred_batch, 1e-8, 1 - 1e-8)
        # COMPUTE ACTOR OFF POLICY LOSS
        p_old_new = y_pred_batch.gather(1, batch_selected_action.unsqueeze(-1))
        #actor_on_loss = -torch.log(y_pred.gather(1, selected_action.unsqueeze(-1))) * (reward + 0.99 * next_expected_reward - expected_reward)
        actor_off_loss = torch.mean(-torch.log(p_old_new) * batch_rewards)
        return actor_off_loss
    
    def off_policy_losses_policy_gradient_cat(self, batch_observation: torch.Tensor, batch_action, batch_selected_action, batch_rewards, batch_next_observation):
        y_pred_batch, _ = self.model(batch_observation)
        correct_actions = F.one_hot(batch_selected_action, num_classes=batch_action.size(1)).float()
        return F.cross_entropy(y_pred_batch, correct_actions)

    def retrain_off_policy(self, batch_size: int = 1):
        """ network learns to improve """
        for shop_name in self.shops:
            observations, actions, selected_actions, rewards, next_observations  = self.replay_buffers[shop_name].get_batch(batch_size, random=not args.disable_random_batch, balanced=args.balanced_sampling, positive=args.positive_sampling)
            if observations is None:
                continue
            self.model.actor_optimizer.zero_grad()
            actor_loss = self.off_policy_losses_policy_gradient(observations, actions, selected_actions, rewards, next_observations)
            actor_loss.backward()
            self.model.actor_optimizer.step()
    
    def retrain_off_policy_cat(self, batch_size: int = 1):
        """ network learns to improve """
        for shop_name in self.shops:
            observations, actions, selected_actions, rewards, next_observations  = self.replay_buffers[shop_name].get_batch(batch_size, random=not args.disable_random_batch, balanced=False, positive=True)
            if observations is None:
                continue
            self.model.actor_optimizer.zero_grad()
            actor_loss = self.off_policy_losses_policy_gradient_cat(observations, actions, selected_actions, rewards, next_observations)
            actor_loss.backward()
            self.model.actor_optimizer.step()

    def learn_off_policy(self, batch_size: int = 1):
        """ network learns to improve """
        metrics = []
        for shop_name in self.shops:
            observations, actions, selected_actions, rewards, next_observations  = self.replay_buffers[shop_name].get_batch(batch_size, random=not args.disable_random_batch, balanced=args.balanced_sampling, positive=args.positive_sampling)
            if observations is None:
                continue
            torch.autograd.set_detect_anomaly(True)
            critic_loss = self.off_policy_critic_loss(observations, actions, selected_actions, rewards, next_observations)
            actor_loss = self.off_policy_actor_loss(observations, actions, selected_actions, rewards, next_observations)
            self.model.critic_optimizer.zero_grad()
            self.model.actor_optimizer.zero_grad()
            actor_loss.backward()
            # critic_loss.backward()
            # self.model.critic_optimizer.step()
            self.model.actor_optimizer.step()

            metrics.append({"actor": float(actor_loss),
                           "critic": float(critic_loss)})
        return metrics

    def on_off_policy_actor_loss(self, observation, selected_action, reward, next_observation, batch_observation, batch_action, batch_selected_action, batch_rewards, batch_next_observation):
        action, _ = self.model(observation)
        p_new, v_new = self.model(batch_observation)
        p_old = batch_action.gather(1, batch_selected_action.unsqueeze(-1))
        p_old_new = p_new.gather(1, batch_selected_action.unsqueeze(-1))
        value_o_t = v_new.detach()
        _, value_o_t1 = self.model(batch_next_observation)
        value_o_t1.detach()

        _, next_expected_reward = self.model(next_observation)
        next_expected_reward = next_expected_reward.detach()
        _, expected_reward = self.model(observation)
        expected_reward = expected_reward.detach()
        on_loss = -torch.log(action.gather(1, selected_action.unsqueeze(-1))) * (reward + 0.99 * next_expected_reward - expected_reward)
        off_loss = -torch.mean((p_old_new / p_old) * torch.log(p_old_new)*(batch_rewards+0.99 * value_o_t1 - value_o_t))
        on_loss,_ = self.on_policy_loss(observation, selected_action, reward)
        return on_loss + args.off_policy_factor * off_loss

    def on_off_policy_critic_loss(self, observation, selected_action, reward, next_observation, batch_observation, batch_action, batch_selected_action, batch_rewards, batch_next_observation):
        p_new, v_new = self.model(batch_observation)
        p_old = batch_action.gather(1, batch_selected_action.unsqueeze(-1)).detach()
        p_old_new = p_new.gather(1, batch_selected_action.unsqueeze(-1)).detach()
        value_o_t = v_new
        _, value_o_t1 = self.model(batch_next_observation)
        y_off = batch_rewards + 0.99 * value_o_t1
        _, expected_reward = self.model(observation)
        _, next_expected_reward = self.model(next_observation)
        y_on = reward + 0.99 * next_expected_reward
        on_loss =  torch.pow(torch.sum(torch.abs(expected_reward - y_on)), 2)
        off_loss = torch.mean(-(p_old_new / p_old) * torch.pow(torch.sum(torch.abs(value_o_t - y_off)), 2))
        _,on_loss = self.on_policy_loss(observation, selected_action, reward)
        return on_loss + args.off_policy_factor * off_loss
    
    def y(self, reward, value):
        return reward + 0.99 * value
    
    def on_off_policy_losses(self, observation, selected_action, reward, next_observation, batch_observation, batch_action, batch_selected_action, batch_rewards, batch_next_observation):
        y_pred, critic_value = self.model(observation)
        y_pred_next, critic_value_next = self.model(next_observation)
        y_pred_batch, critic_value_batch = self.model(batch_observation)
        y_pred_next_batch, critic_value_next_batch = self.model(batch_next_observation)
        delta = reward - critic_value.detach()
        _actions = torch.zeros([1, self.n_actions])
        _actions[:, selected_action] = 1.0
        critic_on_loss = F.mse_loss(critic_value.detach(),
                        reward)
        out = torch.clip(y_pred, 1e-8, 1 - 1e-8)
        log_likelihood = _actions * torch.log(out)
        actor_on_loss = torch.sum(-log_likelihood * delta)

        # COMPUTE CRITIC OFF POLICY LOSS
        p_old = batch_action.gather(1, batch_selected_action.unsqueeze(-1)).detach()
        p_old_new = y_pred_batch.gather(1, batch_selected_action.unsqueeze(-1)).detach()

        y_off = self.y(batch_rewards, critic_value_next)
        critic_off_loss = torch.mean(-(p_old_new / p_old) * torch.pow(torch.sum(torch.abs(critic_value - y_off)), 2))
        
        # COMPUTE ACTOR OFF POLICY LOSS
        p_old = batch_action.gather(1, batch_selected_action.unsqueeze(-1))
        p_old_new = y_pred_batch.gather(1, batch_selected_action.unsqueeze(-1))
        value_o_t = critic_value_batch.detach()
        value_o_t1 = critic_value_next_batch.detach()
        next_expected_reward = critic_value_next.detach()
        expected_reward = critic_value.detach()
        #actor_on_loss = -torch.log(y_pred.gather(1, selected_action.unsqueeze(-1))) * (reward + 0.99 * next_expected_reward - expected_reward)
        actor_off_loss = -torch.mean((p_old_new / p_old) * torch.log(p_old_new)*(batch_rewards+0.99 * value_o_t1 - value_o_t))
        
        return actor_on_loss + args.off_policy_factor * actor_off_loss, critic_on_loss + args.off_policy_factor * critic_off_loss

    def on_off_policy_losses_policy_gradient(self, observation, selected_action, reward, next_observation, batch_observation, batch_action, batch_selected_action, batch_rewards, batch_next_observation):
        y_pred, _ = self.model(observation)
        y_pred = torch.clip(y_pred, 1e-8, 1 - 1e-8)
        y_pred_batch, _ = self.model(batch_observation)
        y_pred_batch = torch.clip(y_pred_batch, 1e-8, 1 - 1e-8)
        _actions = torch.zeros([1, self.n_actions])
        _actions[:, selected_action] = 1.0
        out = torch.clip(y_pred, 1e-8, 1 - 1e-8)
        log_likelihood = _actions * torch.log(out)
        actor_on_loss = torch.sum(-log_likelihood * reward)
        # COMPUTE ACTOR OFF POLICY LOSS
        p_old = batch_action.gather(1, batch_selected_action.unsqueeze(-1))
        p_old_new = y_pred_batch.gather(1, batch_selected_action.unsqueeze(-1))
        #actor_on_loss = -torch.log(y_pred.gather(1, selected_action.unsqueeze(-1))) * (reward + 0.99 * next_expected_reward - expected_reward)
        if args.off_policy_loss_cat:
            indices = torch.where(batch_rewards > 0)[0]
            y_pred_batch_pos = y_pred_batch[indices]
            batch_selected_action_pos = batch_selected_action[indices]
            correct_actions = F.one_hot(batch_selected_action_pos, num_classes=batch_action.size(1)).float()
            actor_off_loss = F.cross_entropy(y_pred_batch_pos, correct_actions)
        else:
            actor_off_loss = torch.mean(-torch.log(p_old_new) * batch_rewards)
        return (1-args.off_policy_factor) * actor_on_loss + args.off_policy_factor * actor_off_loss

    def learn_on_off_policy(self, states, selected_action, reward, states_, dones, batch_size: int = 1):
        """ network learns to improve """
        self.model.train()
        metrics = []
        selected_actions = {
            a["shop"]: torch.tensor(Components.value_list().index(a["component"])) for a in selected_action.values()
        }
        for shop_name in self.shops:
            shop_reward = torch.tensor([reward[0][shop_name]]).unsqueeze(0)
            # if shop_reward[0][0] > 0:
            #     del self.memory[shop_name]
            if shop_name not in selected_actions:
                continue
            shop_selected_action = selected_actions[shop_name].unsqueeze(0)
            shop_state = states[shop_name].encode_to_tensor().unsqueeze(0)
            shop_next_state = states_[shop_name].encode_to_tensor().unsqueeze(0)
            torch.autograd.set_detect_anomaly(True)
            batch_observations, batch_actions, batch_selected_actions, batch_rewards, batch_next_observations  = self.replay_buffers[shop_name].get_batch(batch_size, random=not args.disable_random_batch, balanced=args.balanced_sampling, positive=args.positive_sampling)
            if batch_observations is None:
                # Kein off-policy anteil = reines on-policy
                self.model.actor_optimizer.zero_grad()
                self.model.critic_optimizer.zero_grad()
                actor_loss, critic_loss = self.on_policy_loss(shop_state, shop_selected_action, shop_reward)
                actor_loss.backward()
                self.model.actor_optimizer.step()
                critic_loss.backward()
                self.model.critic_optimizer.step()
                continue
            policy_gradient = True
            if policy_gradient:
                self.model.actor_optimizer.zero_grad()
                actor_loss = self.on_off_policy_losses_policy_gradient(shop_state, shop_selected_action, shop_reward, shop_next_state, batch_observations, batch_actions, batch_selected_actions, batch_rewards, batch_next_observations)
                actor_loss.backward()
                self.model.actor_optimizer.step()
                critic_loss = 0
            else:
                optimized_loss = True
                if optimized_loss:
                    self.model.critic_optimizer.zero_grad()
                    self.model.actor_optimizer.zero_grad()
                    actor_loss, critic_loss = self.on_off_policy_losses(shop_state, shop_selected_action, shop_reward, shop_next_state, batch_observations, batch_actions, batch_selected_actions, batch_rewards, batch_next_observations)
                    actor_loss.backward()
                    self.model.actor_optimizer.step()
                    # critic_loss.backward()
                    # self.model.critic_optimizer.step()
                else:
                    self.model.critic_optimizer.zero_grad()
                    critic_loss = self.on_off_policy_critic_loss(shop_state, shop_selected_action, shop_reward, shop_next_state, batch_observations, batch_actions, batch_selected_actions, batch_rewards, batch_next_observations)
                    critic_loss.backward()
                    #self.model.critic_optimizer.step()
                    self.model.actor_optimizer.zero_grad()
                    actor_loss = self.on_off_policy_actor_loss(shop_state, shop_selected_action, shop_reward, shop_next_state, batch_observations, batch_actions, batch_selected_actions, batch_rewards, batch_next_observations)
                    actor_loss.backward()
                    self.model.actor_optimizer.step()

            metrics.append({"actor": float(actor_loss),
                           "critic": float(critic_loss)})
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

    def reset_sampled_actions_mem(self):
        self.sampled_actions = {}

    def add_to_replay_buffer(self, states, probabilites, actions, reward, next_states, dones):
        for action in actions.values():
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
                self.replay_buffers[shop_name].add(state_tensor, torch.from_numpy(probabilites[shop_name]), torch.tensor(Components.value_list().index(action["component"])), reward_tensor, next_state_tensor)

    def handle_episode_observation(self, sysobservation: SystemObservation):
        for shop_name, observation in sysobservation.shops.items():
            self.replay_buffers[shop_name].update_dist(observation.encode_to_tensor())
