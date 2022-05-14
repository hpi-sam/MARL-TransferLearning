import os

from mrubis_controller.marl.helper import build_reward_plot, build_count_plot, build_loss_plot, get_current_time
from multi_agent_controller import MultiAgentController

# from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
from mrubis_controller.marl.mrubis_env import MrubisEnv


class Runner:
    def __init__(self, args, env, shop_distribution, save_model=False, load_models_data=None,
                 robustness_activated=False):
        self.args = args
        self.shop_distribution = shop_distribution
        self.load_models_data = load_models_data  # if not None saved models are loaded
        self.env = env
        self.mac = MultiAgentController(self.shop_distribution, self.load_models_data, robustness_activated)
        self.t = 0
        self.inner_t = 0
        self.save_model_interval = 100  # interval of saving models
        self.save_model = save_model
        self.base_dir = f"./mrubis_controller/marl/data/logs/{get_current_time()}"

    def reset(self, training_activated=True):
        """ reset all variables and init env """
        self.t = 0
        self.env.reset()
        self.mac.init(self.env.action_space, training_activated)

    def close_env(self):
        self.env.close()

    def run(self, episodes, train=True):
        """ runs the simulation """
        rewards = []
        metrics = []
        self.reset(training_activated=train)
        count_till_fixed = {shop: [] for agent in self.shop_distribution for shop in agent}
        while self.t < episodes:
            terminated = False
            observations = self.env.reset()
            while not terminated:
                self.inner_t += 1
                actions = self.mac.select_actions(observations)

                reward, observations_, terminated, env_info = self.env.step(actions)
                if actions is not None:
                    rewards.append(reward)

                    if train:
                        metrics.append(self.mac.learn(observations, actions, reward, observations_, terminated))
                observations = observations_

                if terminated:
                    self.inner_t = 0
                    for shop, count in env_info['stats'].items():
                        count_till_fixed[shop].append(count)

            self.t += 1
            if self.t % self.save_model_interval == 0:
                if not os.path.exists(self.base_dir):
                    os.makedirs(self.base_dir)
                build_count_plot(self.base_dir, count_till_fixed, self.t, self.shop_distribution)
                if train:
                    self.mac.save_models(self.t)
                    build_reward_plot(self.base_dir, rewards, self.t, self.shop_distribution)
                    build_loss_plot(self.base_dir, metrics, self.t)
            print(f"episode {self.t} done")



if __name__ == "__main__":
    episodes = 400
    # mock_env = MrubisMockEnv(number_of_shops=5, shop_config=[1, 0, False])
    env = MrubisEnv(
        episodes=episodes,
        negative_reward=-1,
        propagation_probability=0.5,
        shops=20,
        injection_mean=5,
        injection_variance=2,
        trace="",
        root_causes="",
        trace_length=0)
    shop_distribution_example = [{'mRUBiS #1'}]
    load_model = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None}
    # load_model = {0: {'start_time': 'test_robustness', 'episode': 500}, 1: None}
    Runner(None, env, shop_distribution_example, save_model=True, load_models_data=load_model,
           robustness_activated=False).run(episodes, train=True)
