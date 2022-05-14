from mrubis_controller.marl.agent import encode_observations
from mrubis_controller.marl.data.data_generator import DataGenerator
from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
import numpy as np


def test_datagenerator_without_dependency_and_offset():
    data_generator = DataGenerator(number_of_shops=3)
    data_generator.set_shop_config([1, 0, False])

    mock_env = MrubisMockEnv()
    mock_env.reset()

    for i in range(10):
        observation, failing_component = data_generator.generate_shops_with_failures(i)
        for shop, component in failing_component.items():
            assert observation[shop][component]['failure_name'] != 'None'
            one_hot_state = encode_observations(observation[shop])[np.newaxis, :][0]
            assert sum(one_hot_state) == 1
            failing_component_from_action_space = list(mock_env.action_space)[np.where(one_hot_state == 1)[0][0]]
            assert failing_component_from_action_space == component
