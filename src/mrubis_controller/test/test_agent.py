from numpy.testing import assert_array_equal
from pathlib import Path

from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
from mrubis_controller.marl.agent import Agent


def init_agent(shops=None):
    if shops is None:
        shops = {'mRUBiS #1'}
    action_space = ['Availability Item Filter', 'Bid and Buy Service', 'Seller Reputation Item Filter',
                    'User Management Service', 'Reputation Service', 'Comment Item Filter',
                    'Recommendation Item Filter', 'Authentication Service', 'Persistence Service',
                    'Item Management Service', 'Buy Now Item Filter', 'Query Service', 'Past Sales Item Filter',
                    'Inventory Service', 'Region Item Filter', 'Category Item Filter', 'Last Second Sales Item Filter',
                    'Future Sales Item Filter']
    episode = 0
    agent = Agent(index=0, shops=shops,
                  action_space_inverted=action_space,
                  load_models_data=None,
                  ridge_regression_train_data_path=Path('./data/TrainingmRUBiS_Theta0.05_NonStationary.csv'))
    return episode, agent


def test_save_and_load_models():
    episode, agent = init_agent()
    agent.save(episode)
    load_model = {"start_time": agent.start_time, "episode": episode}
    actor, critic, policy = agent.load_models(load_model)

    layer = 'layer_last'
    # loaded and saved have equal weights and bias
    assert_array_equal(actor.get_layer(layer).get_weights()[0], agent.actor.get_layer(layer).get_weights()[0])
    assert_array_equal(actor.get_layer(layer).get_weights()[1], agent.actor.get_layer(layer).get_weights()[1])
    assert_array_equal(critic.get_layer(layer).get_weights()[0], agent.critic.get_layer(layer).get_weights()[0])
    assert_array_equal(critic.get_layer(layer).get_weights()[1], agent.critic.get_layer(layer).get_weights()[1])

    # loaded actor and critic share the same layer
    assert actor.get_layer(layer) == critic.get_layer(layer)


def test_choose_action():
    mock_env = MrubisMockEnv(number_of_shops=1, shop_config=[1, 0, False])
    observation = mock_env.reset()
    episode, agent = init_agent()
    actions = agent.choose_action(observation)
    assert actions[0]['shop'] == 'mRUBiS #1'
    assert actions[0]['component']
    assert actions[0]['predicted_utility']
