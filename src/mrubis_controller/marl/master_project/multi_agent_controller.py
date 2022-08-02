from pathlib import Path

from marl.master_project.agent import Agent
from marl.master_project.mrubis_data_helper import build_observations, build_rewards, build_actions
from marl.master_project.rank_learner import RankLearner
from marl.master_project.robustness_component import RobustnessComponent


class MultiAgentController:
    def __init__(self, load_models_data, robustness_activated=False):
        # list of named shops per agent identified by the index
        self.shop_distribution = [{'mRUBiS #1', 'mRUBiS #2', 'mRUBiS #3', 'mRUBiS #4', 'mRUBiS #5',
                                   'mRUBiS #6', 'mRUBiS #7', 'mRUBiS #8', 'mRUBiS #9', 'mRUBiS #10'}]
        self.load_models_data = load_models_data
        self.rank_learner = RankLearner(1, None)
        self.agents = None
        self.ridge_regression_train_data_path = Path(
            './data/TrainingmRUBiS_Theta0.05_NonStationary.csv')
        self.robustness = RobustnessComponent(len(self.shop_distribution))
        self.robustness_activated = robustness_activated

    def init(self, action_space, training_activated):
        self._build_agents(action_space, training_activated)

    def select_actions(self, observations):
        """ based on observations select actions
            each agent must be called to determine the actions per agent
            the rank learner has to be called to sort those actions
            returns a sorted list of actions
        """
        if self.robustness_activated:
            self.robustness.plan(self.agents)

        actions = []
        regrets = {}
        root_causes = {}
        action_probabilities = {}
        for index, agent in enumerate(self.agents):
            if self.robustness_activated and self.robustness.skip_agent(index):
                continue
            challenged_shops = self.robustness.get_execution_plan(
                index) if self.robustness_activated else None
            action, regret, root_cause, probabilities = agent.choose_action(
                build_observations(self.agents, index, observations, challenged_shops))
            regrets[index] = regret
            action_probabilities = {**action_probabilities, **probabilities}
            root_causes[index] = root_cause
            actions.append(action)
        return self.rank_learner.sort_actions(actions), regrets, root_causes, action_probabilities

    def add_to_replay_buffer(self, observations, probabilities, actions, rewards, observations_, dones):
        for index, agent in enumerate(self.agents):
            agent.add_to_replay_buffer(build_observations(self.agents, index, observations),
                                        probabilities,
                                       build_actions(self.agents, index, actions),
                                       build_rewards(self.agents, index, rewards),
                                       build_observations(self.agents, index, observations_),
                                       dones)

    def learn_off_policy(self, batch_size):
        """ start learning for Agents and RankLearner """
        metrics = {}
        for index, agent in enumerate(self.agents):
            metrics[index] = agent.learn_off_policy(batch_size)
        return metrics
    
    def learn_on_policy(self, observations, actions, rewards, observations_, dones):
        """ start learning for Agents and RankLearner """
        metrics = {}
        for index, agent in enumerate(self.agents):
            if self.robustness_activated and self.robustness.skip_agent(index):
                continue

            if self.robustness_activated:
                self.robustness.validate_execution(
                    self.agents, index, observations_)

            metrics[index] = agent.learn_on_policy(build_observations(self.agents, index, observations),
                                         build_actions(
                                             self.agents, index, actions),
                                         build_rewards(
                                             self.agents, index, rewards),
                                         build_observations(
                                             self.agents, index, observations_),
                                         dones)

        history = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'observations_': observations_,
            'dones': dones
        }
        self.robustness.monitor(metrics, history)

        if self.robustness_activated:
            self.robustness.analyze(self.agents)

        return metrics
    
    def learn_on_off_policy(self, observations, actions, rewards, observations_, dones, batch_size):
        """ start learning for Agents and RankLearner """
        metrics = {}
        for index, agent in enumerate(self.agents):
            if self.robustness_activated and self.robustness.skip_agent(index):
                continue

            if self.robustness_activated:
                self.robustness.validate_execution(
                    self.agents, index, observations_)

            metrics[index] = agent.learn_on_off_policy(build_observations(self.agents, index, observations),
                                         build_actions(
                                             self.agents, index, actions),
                                         build_rewards(
                                             self.agents, index, rewards),
                                         build_observations(
                                             self.agents, index, observations_),
                                         dones,
                                         batch_size)

        history = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'observations_': observations_,
            'dones': dones
        }
        self.robustness.monitor(metrics, history)

        if self.robustness_activated:
            self.robustness.analyze(self.agents)

        return metrics

    def save_models(self, episode):
        """ save models of agents and rank learner """
        for agent in self.agents:
            agent.save(episode)

    def _build_agents(self, action_space, training_activated):
        """ based on shop distribution the agents will be initialized """
        self.agents = [
            Agent(
                shops=shops,
                action_space_inverted=action_space,
                load_models_data=self.load_models_data[index],
                ridge_regression_train_data_path=self.ridge_regression_train_data_path,
                index=index,
                training_activated=training_activated
            )
            for index, shops in enumerate(self.shop_distribution)
        ]
