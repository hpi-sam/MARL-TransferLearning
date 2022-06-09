import os

from marl.helper import build_reward_plot, build_count_plot, build_loss_plot, get_current_time, \
    build_regret_plot, print_logs
# from marl.mrubis_mock_env import MrubisMockEnv
from marl.mrubis_env import MrubisEnv
from marl.multi_agent_controller import MultiAgentController


class Runner:
    def __init__(self, args, env: MrubisEnv, shop_distribution, save_model=False, load_models_data=None,
                 robustness_activated=False, training_activated=True):
        self.args = args
        self.shop_distribution = shop_distribution
        self.load_models_data = load_models_data  # if not None saved models are loaded
        self.env = env
        self.mac = MultiAgentController(self.shop_distribution, self.load_models_data, robustness_activated)
        self.t = 0
        self.inner_t = 0
        self.save_model_interval = 100  # interval of saving models
        self.save_model = save_model
        self.training_activated = training_activated
        self.base_dir = f"./marl/data/logs/{get_current_time()}"

    def reset(self):
        """ reset all variables and init env """
        self.t = 0
        self.env.reset()
        self.mac.init(self.env.action_space, self.training_activated)

    def close_env(self):
        self.env.close()

    def run(self, episodes):
        """ runs the simulation """
        rewards = []
        metrics = []
        regrets = []
        logs = []
        self.reset()
        count_till_fixed = {shop: [] for agent in self.shop_distribution for shop in agent}
        while self.t < episodes:
            terminated = False
            observations = self.env.reset()
            regrets.append({})
            logs.append([])
            while not terminated:
                logs[self.t].append({})
                actions, regret, root_cause = self.mac.select_actions(observations)
                # if regret is not None:
                #     regrets.append(regret)
                reward, observations_, terminated, env_info = self.env.step(actions)
                if actions is not None:
                    rewards.append(reward)
                    metrics.append(self.mac.learn(observations, actions, reward, observations_, terminated))

                    for index in regret:
                        if index not in regrets[self.t]:
                            regrets[self.t][index] = {}
                            for shop in regret[index]:
                                regrets[self.t][index][shop] = {}
                        for shop in regret[index]:
                            regrets[self.t][index][shop] = regret[index][shop]

                    logs[self.t][self.inner_t] = {}
                    for cause in root_cause.values():
                        for individual_shop, individual_cause in cause.items():
                            logs[self.t][self.inner_t][individual_shop] = {}
                            logs[self.t][self.inner_t][individual_shop]['Root causes'] = individual_cause

                    for action in actions.values():
                        logs[self.t][self.inner_t][action['shop']]['Actions'] = action['component']

                    for shop in reward[0]:
                        if shop in logs[self.t][self.inner_t]:
                            logs[self.t][self.inner_t][shop]['Reward'] = reward[0][shop]

                observations = observations_
                self.inner_t += 1

                if terminated:
                    self.inner_t = 0
                    for shop, count in env_info['stats'].items():
                        count_till_fixed[shop].append(count)

            self.t += 1
            if self.t % self.save_model_interval == 0:
                self.mac.save_models(self.t)
                if not os.path.exists(self.base_dir):
                    os.makedirs(self.base_dir)
                build_count_plot(self.base_dir, count_till_fixed, self.t, self.shop_distribution)
                build_reward_plot(self.base_dir, rewards, self.t, self.shop_distribution)
                build_loss_plot(self.base_dir, metrics, self.t)
                build_regret_plot(self.base_dir, regrets, self.t, self.shop_distribution, self.training_activated, avg_over_agents=True)
                print_logs(self.base_dir, logs)
            print(f"episode {self.t} done")

    def create_prob_distribution(self):
        self.reset(training_activated=False)
        observations = self.env.reset()
        actions = self.mac.select_actions(observations)
        reward, observations_, terminated, env_info = self.env.step(actions)
        # TODO build probability plot


if __name__ == "__main__":
    episodes = 400
    # mock_env = MrubisMockEnv(number_of_shops=5, shop_config=[1, 0, False])
    env = MrubisEnv(
        episodes=episodes,
        negative_reward=-1,
        propagation_probability=0.5,
        shops=10,
        injection_mean=5,
        injection_variance=2,
        trace="",
        trace_length=0,
        send_root_issue=True,
        reward_variance=5)
    shop_distribution_example = [{'mRUBiS #1', 'mRUBiS #2', 'mRUBiS #3', 'mRUBiS #4', 'mRUBiS #5',
                                 'mRUBiS #6', 'mRUBiS #7', 'mRUBiS #8', 'mRUBiS #9', 'mRUBiS #10'}]
    load_model = {0: None, 1: None, 2: None, 3: None, 4: None,
                  5: None, 6: None, 7: None, 8: None, 9: None,
                  10: None, 11: None, 12: None, 13: None, 14: None,
                  15: None, 16: None, 17: None, 18: None, 19: None,
                  20: None, 21: None, 22: None, 23: None, 24: None,
                  25: None, 26: None, 27: None, 28: None, 29: None,
                  30: None, 31: None, 32: None, 33: None, 34: None,
                  35: None, 36: None, 37: None, 38: None, 39: None,
                  40: None, 41: None, 42: None, 43: None, 44: None,
                  45: None, 46: None, 47: None, 48: None, 49: None,
                  50: None, 51: None, 52: None, 53: None, 54: None,
                  55: None, 56: None, 57: None, 58: None, 59: None,
                  60: None, 61: None, 62: None, 63: None, 64: None,
                  65: None, 66: None, 67: None, 68: None, 69: None,
                  70: None, 71: None, 72: None, 73: None, 74: None,
                  75: None, 76: None, 77: None, 78: None, 79: None}
    # load_model = {0: {'start_time': 'trace_experiments_length_5', 'episode': 300}}
    Runner(None, env, shop_distribution_example, save_model=True, load_models_data=load_model,
           robustness_activated=False, training_activated=True).run(episodes)
