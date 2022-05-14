from pathlib import Path

from mrubis_controller.marl.agent import Agent
from mrubis_controller.marl.random_agent import RandomAgent
from mrubis_controller.marl.mrubis_data_helper import build_observations, build_rewards, build_actions
from mrubis_controller.marl.rank_learner import RankLearner
from mrubis_controller.marl.robustness_component import RobustnessComponent


class MultiAgentController:
    def __init__(self, shop_distribution, load_models_data, robustness_activated):
        # list of named shops per agent identified by the index
        self.shop_distribution = shop_distribution
        self.load_models_data = load_models_data
        self.rank_learner = RankLearner(1, None)
        self.agents = None
        self.ridge_regression_train_data_path = Path('./data/TrainingmRUBiS_Theta0.05_NonStationary.csv')
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
        for index, agent in enumerate(self.agents):
            if self.robustness_activated and self.robustness.skip_agent(index):
                continue
            challenged_shops = self.robustness.get_execution_plan(index) if self.robustness_activated else None
            actions.append(agent.choose_action(build_observations(self.agents, index, observations, challenged_shops)))

        return self.rank_learner.sort_actions(actions)

    def learn(self, observations, actions, rewards, observations_, dones):
        """ start learning for Agents and RankLearner """
        metrics = {}
        for index, agent in enumerate(self.agents):
            if self.robustness_activated and self.robustness.skip_agent(index):
                continue

            if self.robustness_activated:
                self.robustness.validate_execution(self.agents, index, observations_)

            metrics[index] = agent.learn(build_observations(self.agents, index, observations),
                                         build_actions(self.agents, index, actions),
                                         build_rewards(self.agents, index, rewards),
                                         build_observations(self.agents, index, observations_),
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
