from mrubis_controller.jakobs_model.component_utility_predictor import RidgeUtilityPredictor
import pandas as pd
from entities.observation import ComponentObservation, ShopObservation, SystemObservation


class AgentActionSorter:
    def __init__(self, train_data_path):
        self.utility_model = RidgeUtilityPredictor()
        # Train the model on the provided batch file
        self.utility_model.load_train_data(train_data_path)
        self.utility_model.train_on_batch_file()

    def predict_optimal_utility_of_fixed_components(self, action, observation):
        """Predict the optimal utility of a component which should be fixed"""
        observation = observation.components[action['component']].encode_to_dict(
        )
        return self.utility_model.predict_on_mrubis_output(pd.DataFrame(observation, index=[0]))[0]
