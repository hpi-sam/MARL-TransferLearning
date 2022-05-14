import gym
import logging

import json
from time import sleep
from subprocess import PIPE, Popen

from chunkedsocketcommunicator import ChunkedSocketCommunicator
from mrubis_controller.marl.mrubis_data_helper import get_current_utility

logging.basicConfig()
logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)


class MrubisEnv(gym.Env):
    def __init__(self, json_path='path.json', external_start=True, episodes=500, negative_reward=-1,
                 propagation_probability=0.5, shops=10, injection_mean=6, injection_variance=2, trace="",
                 root_causes = "", trace_length = 0):
        super(MrubisEnv, self).__init__()
        self.launch_args = None
        self.action_space = None
        self.observation_space = None
        self.communicator = None
        self.prior_utility = None  # used for calculation of diff as a reward
        self.t = 0
        self.termination_t = 3
        self.inner_t = 0
        self.stats = {}
        self.episodes = episodes
        self.terminated = False
        self.negative_reward = negative_reward
        self.shops = shops
        self.propagation_probability = propagation_probability
        self.injection_mean = injection_mean
        self.injection_variance = injection_variance
        self.trace = trace
        self.root_caues = root_causes
        self.trace_length = trace_length

        '''Create a new instance of the mRUBiS environment class'''
        self.external_start = external_start

        # Put your command line here (In Eclipse: Run -> Run Configurations... -> Show Command Line)
        # with open(json_path, 'r') as f:
        #     variable_paths = json.load(f)

        '''
        self.launch_args = [
            variable_paths['java_path'],
            '-DFile.encoding=UTF-8',
            '-classpath',
            variable_paths['dependency_paths'],
            '-XX:+ShowCodeDetailsInExceptionMessages',
            'mRUBiS_Tasks.Task_1',
        ]
        '''

        self.run_counter = 0
        self.number_of_shops = 0
        self.number_of_issues_per_shop = {}
        self.current_issue = {}
        self.mrubis_state = {}
        self.mrubis_state_history = []
        self.fix_history = []
        self.current_fixes = None
        self.mrubis_process = None
        self.communicator = None
        self.observation = None

        # initial state and action
        self.reset()

    def step(self, actions):
        """ Returns observation, reward, terminated, info """
        self.inner_t += 1
        if actions is not None:
            self.communicator.println(json.dumps(actions))
        else:
            self.communicator.println(json.dumps({}))
        message = self.communicator.readln()
        assert message == "received"
        self.observation = self._get_state()

        _reward = self._get_reward(self.observation)

        for shop in _reward[0]:
            if _reward[0][shop] > 0:
                _reward[0][shop] = 17
                self.stats[shop] = self.inner_t

        info = self._info()
        if actions is None or self._is_fixed():
            self.t += 1
            self.inner_t = 0
            self.stats = self._init_stats()
            self.terminated = True

        return _reward, self.observation, self.terminated, info

    def reset(self):
        """ Returns initial observations and states """
        if self.communicator is None:
            self.communicator = ChunkedSocketCommunicator()

        if not self.external_start:
            self._start_mrubis()
            if self.mrubis_process.poll() is None:
                logger.info('MRUBIS is running')

        # Account for Java being slow to start on some systems
        sleep(0.5)

        self.t = 0
        self._reset_mrubis()
        self.observation = self._get_state()
        self.prior_utility = get_current_utility(self.observation)
        self.action_space = [components for shops, components in self.observation.items()][0].keys()
        self.stats = self._init_stats()
        self.terminated = False
        return self.observation

    def render(self):
        """ Renders the environment. """
        raise NotImplementedError

    def close(self):
        """ Override close in your subclass to perform any necessary cleanup. """
        self.communicator.send_exit_message()
        self.communicator.close_socket()

        if not self.external_start:
            self._stop_mrubis()

    def seed(self):
        """ Sets the seed for this env's random number generator(s). """
        raise NotImplementedError

    def last(self):
        """ returns last state, reward, terminated, info """
        self.observation = self._get_state()
        return self.observation, self._get_reward(self.observation), self.terminated, self._info()

    def _is_fixed(self):
        for shop in self.observation:
            for component in self.observation[shop]:
                if self.observation[shop][component]["failure_name"] != "None":
                    return False
        return True

    def _get_reward(self, observation):
        """ returns the extracted reward per shop
        """
        current_utility = get_current_utility(observation)
        # print(current_utility)
        diff_utility = {shop: current_utility[shop] - utility for shop, utility in self.prior_utility.items()}
        self.prior_utility = current_utility
        system_utility = sum(diff_utility.values())
        return diff_utility, system_utility

    def _terminated(self):
        return self.t == self.termination_t

    def _start_mrubis(self):
        """Launch mRUBiS as a subprocess. NOTE: Unstable. Manual startup from Eclipse is more robust."""
        self.mrubis_process = Popen(
            self.launch_args,
            stdin=PIPE,
            stdout=PIPE,
            shell=False,
            cwd="../mRUBiS/ML_based_Control"
        )

    def _reset_mrubis(self):
        # self.communicator.println("reset")
        self.communicator.println(json.dumps({
            "reset": str(True),
            "episodes": str(self.episodes),
            "negative_reward": str(self.negative_reward),
            "propagation_probability": str(self.propagation_probability),
            "shops": str(self.shops),
            "injection_mean": str(self.injection_mean),
            "injection_variance": str(self.injection_variance),
            "trace": self.trace,
            "root_causes": self.root_caues,
            "trace_length": str(self.trace_length)
        }))
        response = self.communicator.readln()
        if response == "resetting":
            return True
        else:
            return False

    def _stop_mrubis(self):
        """Terminate the mRUBiS process"""
        self.mrubis_process.terminate()

    def _init_stats(self):
        stats = {}
        for shop in self.observation:
            stats[shop] = -1
        return stats

    def _info(self):
        return {'t': self.t, 'stats': self.stats}

    def _get_state(self):
        '''Query mRUBiS for the state'''
        return self.communicator.get_from_mrubis(json.dumps({"get_state": str(True)}))
