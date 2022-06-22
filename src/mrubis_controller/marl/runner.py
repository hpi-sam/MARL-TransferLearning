import wandb
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
                    # exit()
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

class ReplayBufferRunner:
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
                    # exit()
                    valid_actions = filter(lambda x: has_shop_remaining_issues(current_observations, x.action.shop), list(actions.values()))
                    sendable_actions = list(map(lambda x: x.action.to_sendable_json(), valid_actions))
                    sendable_actions = {i: e for i, e in enumerate(sendable_actions)}
                    reward, next_observations, terminated, env_info = self.env.step(sendable_actions)
                    # self.agent_controller.learn(actions, reward, next_observations)
                    self.agent_controller.add_to_replaybuffer(actions, reward, next_observations)
                    self.agent_controller.learn_from_replaybuffer(100)
                    current_observations = next_observations
                    if terminated:
                        for shop, count in env_info['stats'].items():
                            print(f"{shop} fixed after: {count}")
                            if count != -1:
                                wandb.log({f"{shop}_fixed": count}, step=step)
                        self.env.reset()
                print(f"episode {episode} done")
