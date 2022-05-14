from mrubis_controller.jakobs_model.component_utility_predictor import RidgeUtilityPredictor
import pandas as pd


class AgentActionSorter:
    def __init__(self, train_data_path):
        self.utility_model = RidgeUtilityPredictor()
        # Train the model on the provided batch file
        self.utility_model.load_train_data(train_data_path)
        self.utility_model.train_on_batch_file()

    def predict_optimal_utility_of_fixed_components(self, action, observation):
        """Predict the optimal utility of a component which should be fixed"""
        return self.utility_model.predict_on_mrubis_output(pd.DataFrame(observation[action['component']], index=[0]))[0]
