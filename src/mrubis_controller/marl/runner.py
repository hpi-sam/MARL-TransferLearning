import wandb
from entities.observation import SystemObservation
from marl.mrubis_data_helper import has_shop_remaining_issues
from marl.mrubis_env import MrubisEnv
from marl.shop_agent_controller import ShopAgentController
import alive_progress as ap

class Runner:
    def __init__(self, env: MrubisEnv, agent_controller: ShopAgentController):
        self.env = env
        self.agent_controller = agent_controller # TODO create agent controller

    def reset(self):
        """ reset all variables and init env """
        self.env.reset()
        # TODO reset agent controller

    def close_env(self):
        self.env.close()

    def run(self, episodes):
        """ runs the simulation """
        self.reset()
        step = 0
        with ap.alive_bar(title="Running") as bar:
            for episode in range(episodes):
                wandb.log({'episode': episode}, commit=False, step=step)
                terminated = False
                current_observations = self.env.reset()
                while not terminated:
                    bar()
                    bar.text = f"Episode: {episode} Step: {step}"
                    step += 1
                    actions = self.agent_controller.choose_actions(current_observations)
                    # for action in actions.values():
                    #     print(action.action_tensor.max().item(), action.action_tensor.argmax().item(), end=' ')
                    # print()
                    valid_actions = filter(lambda x: has_shop_remaining_issues(current_observations, x.action.shop), list(actions.values()))
                    sendable_actions = list(map(lambda x: x.action.to_sendable_json(), valid_actions))
                    sendable_actions = {i: e for i, e in enumerate(sendable_actions)}
                    reward, next_observations, terminated, env_info = self.env.step(sendable_actions)
                    self.agent_controller.learn(actions, reward, next_observations)
                    current_observations = next_observations
                    if terminated:
                        for shop, count in env_info['stats'].items():
                            print(f"{shop} fixed after: {count}")
                            if count != -1:
                                wandb.log({f"{shop}_fixed": count}, step=step)
                        self.env.reset()
                print(f"episode {episode} done")

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
    shops = {'mRUBiS #1', 'mRUBiS #2', 'mRUBiS #3', 'mRUBiS #4', 'mRUBiS #5', 'mRUBiS #6', 'mRUBiS #7', 'mRUBiS #8', 'mRUBiS #9', 'mRUBiS #10'}
    # load_model = {0: {'start_time': 'trace_experiments_length_5', 'episode': 300}}
    Runner(env).run(episodes)
