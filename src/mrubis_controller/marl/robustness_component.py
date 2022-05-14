import numpy as np

from mrubis_controller.marl.agent import Agent
from mrubis_controller.marl.mrubis_data_helper import build_replay_buffer
from mrubis_controller.marl.mrubis_data_helper import has_shop_remaining_issues


class RobustnessComponent:
    def __init__(self, number_of_agents, config=None):
        if config is None:
            self.config = {
                'actor_lower_threshold': 0,  # value against which the specified statistic is compared
                'actor_upper_threshold': 0.75,  # value against which the specified statistic is compared
                'critic_lower_threshold': 5,  # value against which the specified statistic is compared
                'critic_upper_threshold': 15,  # value against which the specified statistic is compared
                'evaluation_periods': 5,  # number of periods over which data is compared to the specified threshold
                'datapoints_to_alarm': 3,
                # number of data points within the Evaluation Periods that must be breaching to cause the alarm
                'retrain_episodes': 100,
            }

        self.agents_status = {index: 'INSUFFICIENT_DATA' for index in range(number_of_agents)}
        # only copy agents that have been OK
        self.previously_stable_agents = {index: False for index in range(number_of_agents)}
        # OK –                  The metric or expression is within the defined threshold.
        # ALARM –               The metric or expression is outside of the defined threshold.
        # INSUFFICIENT_DATA –   The alarm has just started, the metric is not available,
        #                       or not enough data is available for the metric to determine the alarm state.
        # CALIBRATION           The agent is currently trying to recalibrate due to a prior ALARM status
        # RETRAIN               The agent is trying to improve
        # RETIRED               The agent has no shops left and has no tasks to do and is waiting for challenges

        self.calibration_distribution = {}  # stores the ranked shop distribution during calibration mode
        self.retrain_count = {}  # counter for retraining episodes
        self.metrics = []  # store metrics of agent learning
        self.replay_buffer = []  # store history

    def validate_execution(self, agents, agent_index, observations_):
        """
        recheck if calibration was correct or wrong
        if calibration was correct: move shops to other agents by updating shop_distribution
        and learn action for new agent. If calibration was wrong don't learn this and skip
        update calibration_distribution and test next agent
        """

        if self._is_agent_in_charge_for_other_calibrating_agent(agent_index):
            # check if other agent was doing a better job
            failing_agent_shops = self._get_shops_of_agent_in_charge_for_other_calibrating_agent(agent_index)
            for n in range(len(failing_agent_shops['failing_agent_index'])):
                challenged_shops = failing_agent_shops['shops'][n].difference(agents[agent_index].shops)
                for shop in challenged_shops:
                    if has_shop_remaining_issues(observations_, shop):
                        # remove current challenge and wait for next agent to solve this challenge
                        self.calibration_distribution[failing_agent_shops['failing_agent_index'][n]].popitem()

                        # check if all agents were not able to help the agent
                        if len(self.calibration_distribution[
                                   failing_agent_shops['failing_agent_index'][n]].values()) == 0:
                            if self.previously_stable_agents[failing_agent_shops['failing_agent_index'][n]]:
                                self._copy_agent(agents, failing_agent_shops['failing_agent_index'][n])
                            self.agents_status[failing_agent_shops['failing_agent_index'][n]] = 'RETRAIN'
                            self.retrain_count[failing_agent_shops['failing_agent_index'][n]] = 0
                    else:
                        self._move_shops(agents, failing_agent_shops['failing_agent_index'][n], agent_index,
                                         challenged_shops)

    def get_execution_plan(self, agent_index):
        """ this is the EXECUTE stage for the challenge of the agents
            If the agent is next in line to test his strength for helping a shop, this method returns the shops of that
            other agent and its own shops. If there is no challenge available, None is returned
        """
        if self.agents_status[agent_index] == 'CALIBRATION':
            # don't use an agent that is currently in calibration mode
            return
        elif self._is_agent_in_charge_for_other_calibrating_agent(agent_index):
            # get next challenge
            failing_agent_shops = self._get_shops_of_agent_in_charge_for_other_calibrating_agent(agent_index)
            return set.union(*failing_agent_shops['shops'])
        else:
            return None

    def skip_agent(self, index):
        """ skip agents that are in CALIBRATION or ALARM and skip retired agents only if there are not challenged """
        return self.agents_status[index] in ['CALIBRATION', 'ALARM'] or (
                self.agents_status[index] == 'RETIRED' and self._is_agent_in_charge_for_other_calibrating_agent(index))

    def monitor(self, metrics, history):
        """ this is the MONITOR stage that saves metrics and history """
        self.replay_buffer.append(history)
        self.metrics.append(metrics)

    def analyze(self, agents):
        """ this is the ANALYSIS state that checks whether metrics have breached a threshold and trigger an alarm
        """
        eval_periods = self.config['evaluation_periods']
        for index, agent in enumerate(agents):
            if self.agents_status[index] in ['ALARM', 'CALIBRATION', 'RETIRED']:
                continue
            if self.agents_status[index] in ['RETRAIN']:
                # after 'retrain_episodes' steps, the agent should move to OK or ALARM to enable validation again
                self.retrain_count[index] += 1
                if self.retrain_count[index] < self.config['retrain_episodes']:
                    continue
                else:
                    # reset retrain count
                    # status will be set to ALARM or OK following the code below
                    self.retrain_count[index] = 0
            agent_metric = [agent_metric for metric in self.metrics for agent_metric in metric[index]][-eval_periods:]
            if len(agent_metric) < eval_periods:
                self.agents_status[index] = 'INSUFFICIENT_DATA'
            else:
                # check if any NN is breaching the defined threshold
                # at least datapoints_to_alarm steps must breach the threshold to set an ALARM
                # otherwise the status is OK
                # TODO: why does the critic loss change by changing the data generator?
                for network in ['actor']:
                    breached_metrics = [item[network] for item in agent_metric if
                                        self.config[f"{network}_lower_threshold"] > abs(item[network]) or
                                        abs(item[network]) > self.config[f"{network}_upper_threshold"]]
                    if len(breached_metrics) >= self.config['datapoints_to_alarm']:
                        self.agents_status[index] = 'ALARM'
                        break
                    else:
                        self.agents_status[index] = 'OK'
                        # agent that was OK at some point can be copied before retrained
                        self.previously_stable_agents[index] = True

    def plan(self, agents):
        """ before selecting an action all agents have to be checked if they are in ALARM and init the calibration """
        for index, agent in enumerate(agents):
            if self.agents_status[index] == 'ALARM':
                self._init_calibration_distribution(index, agent, agents)

    def _init_calibration_distribution(self, failing_agent_index, failing_agent, agents):
        """ this is the PLANNING stage
            as soon as an agent is in alarm all available agents (in state OK) that do not fail a pre-check
            (don't make the same last mistake), are ranked by a challenge (last observation).
            The challenge is used to rank the agents by the lowest critic loss which should help to find the best agent
            faster. The execution plan is stored inside the calibration_distribution if any agent is left that did pass
            the pre-checks. Otherwise, the agent has to be retrained as no agent is available to help that agent
        """
        relevant_history = build_replay_buffer(agents, failing_agent_index, self.replay_buffer)
        agents_predictions = {}
        for index, agent in enumerate(agents):
            if agent != failing_agent:
                # ignore agents in alarm or calibration state
                if self.agents_status[index] not in ['OK']:
                    continue

                # ignore agents similar to failing agent
                if not self._is_agent_check_plausible(agent, relevant_history):
                    continue

                # calc predictions for last step by all agents
                agent_predictions = {}
                for shop_name, components in relevant_history['observations'].items():
                    probabilities = agent.get_probabilities_for_observations(components)
                    max_value = np.amax(probabilities)
                    agent_predictions[shop_name] = max_value
                agents_predictions[index] = agent_predictions

        # rank possible agents bei certainty of action probability
        ranked_agents = self._rank_agents_predictions(relevant_history['observations'], agents_predictions)

        # init calibration_distribution
        ranked_test_distributions = {
            agent_index: {
                'shops': agents[agent_index].shops.union(agents[failing_agent_index].shops)
            }
            for agent_index in ranked_agents.keys()
        }

        if not ranked_test_distributions:
            # no agent is available to start a calibration so retrain that agent
            self.agents_status[failing_agent_index] = 'RETRAIN'
            self.retrain_count[failing_agent_index] = 0
        else:
            self.calibration_distribution[failing_agent_index] = ranked_test_distributions
            self.agents_status[failing_agent_index] = 'CALIBRATION'

    def _is_agent_in_charge_for_other_calibrating_agent(self, index):
        """
        checks if index_agent is next in line and has currently a failing component so the agent can have a challenge
        """
        return any(len(distribution.keys()) > 0 and index == list(distribution.keys())[0] for distribution in
                   self.calibration_distribution.values())

    def _get_shops_of_agent_in_charge_for_other_calibrating_agent(self, agent_index):
        """
        get all distributions if agent_index is next in line and the failing agent has shops with failures
        so we can have challenge for that agent, otherwise take the next agent
        """
        # TODO: return only shops that have remaining issues
        calibration_distribution = {'shops': [], 'failing_agent_index': []}
        for failing_agent_index, distribution in self.calibration_distribution.items():
            if len(distribution.keys()) > 0 and agent_index == list(distribution.keys())[0]:
                calibration_distribution['shops'].append(list(distribution.values())[0]['shops'])
                calibration_distribution['failing_agent_index'].append(failing_agent_index)
        return calibration_distribution or None

    def _rank_agents_predictions(self, observations, agents_predictions):
        """ ranks the agent for a given observation by checking how certain they are to have the correct action """
        ranking_per_shop = {}
        # each agent gets points per ranking per shop
        # the less points the better
        score_per_agent = {agent_index: 0 for agent_index, _ in agents_predictions.items()}  # init scores
        for shop_name, components in observations.items():
            # prepare data for sorting
            max_prediction_per_agent = [{'agent_index': agent_index, 'amax_prediction': predictions[shop_name]} for
                                        agent_index, predictions in agents_predictions.items()]
            # sort predictions
            ranking_per_shop[shop_name] = sorted(max_prediction_per_agent, key=lambda d: d['amax_prediction'],
                                                 reverse=True)
        for shop, ranked_predictions in ranking_per_shop.items():
            for index, prediction in enumerate(ranked_predictions):
                score_per_agent[prediction['agent_index']] += index
        return dict(sorted(score_per_agent.items(), key=lambda item: item[1]))

    def _is_agent_check_plausible(self, agent, relevant_history):
        """ filter agents which would have chosen the exact same action that was wrong """
        actions = agent.choose_action(relevant_history['observations'])
        actions = {action['shop']: action['component'] for action in actions}
        if any(actions[historic_action['shop']] == historic_action['component'] for historic_action in
               relevant_history['actions'].values()):
            return False
        return True

    def _move_shops(self, agents, source_index, destination_index, shops):
        """ moves shops from source agent to destination
            if no shop is left in source agent, that agent retires
        """
        agents[source_index].remove_shops(shops)
        agents[destination_index].add_shops(shops)

        # check if agent has any shops left
        if len(agents[source_index].shops) == 0:
            self.agents_status[source_index] = 'RETIRED'

    def _copy_agent(self, agents, agent_index):
        """ copy agent and retire copy of that agent
            retired agent is a snapshot of a shop configuration that could eventually be relevant in the future
            e.g. a rollback of a shop
        """
        agents[agent_index].save('copy')
        load_model = {'start_time': agents[agent_index].start_time, 'episode': 'copy', 'index': agent_index}
        copy_index = len(agents)
        copy_agent = Agent(
            copy_index,
            {},
            agents[agent_index].action_space_inverted,
            load_model,
            agents[agent_index].ridge_regression_train_data_path,
        )
        copy_agent.start_time = agents[agent_index].start_time

        # append new agent
        agents.append(copy_agent)

        # update status
        self.agents_status[copy_index] = 'RETIRED'

        # setting this to false prevents to copy agent each retrain loop
        self.previously_stable_agents[agent_index] = False
        self.previously_stable_agents[copy_index] = False
