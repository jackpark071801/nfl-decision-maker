import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import os

class WinProbabilityPredictor:
    def __init__(self, config_path):
        # Load the configuration JSON
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        # Paths and variables from the config
        model_save_folder = self.config['output']['model_save_folder']
        scaler_path = os.path.join(model_save_folder, self.config['output']['scaler_path'])
        nn_model_path = os.path.join(model_save_folder, self.config['output']['nn_model_path'])
        linear_model_path = os.path.join(model_save_folder, self.config['output']['linear_model_path'])
        logistic_model_path = os.path.join(model_save_folder, self.config['output']['logistic_model_path'])

        # Load models and scaler
        self.scaler = joblib.load(scaler_path)
        self.nn_model = load_model(nn_model_path)
        self.linear_model = joblib.load(linear_model_path)
        self.logistic_model = joblib.load(logistic_model_path)

        # Retrieve feature variables and interaction settings from the config
        self.feature_variables = self.config['model']['feature_variables']
        self.interaction_vars = self.config['model']['interaction_terms']
        self.include_all_interactions = self.config['model']['include_all_interactions']

    def predict(self, input_data):
        # Create interaction terms if specified in the config
        interaction_terms = []
        if self.include_all_interactions:
            for col1 in input_data.columns:
                for col2 in input_data.columns:
                    if col1 != col2:
                        interaction_term = input_data[col1].values * input_data[col2].values
                        interaction_terms.append(interaction_term)
        else:
            for col1 in self.interaction_vars:
                for col2 in self.interaction_vars:
                    if col1 != col2:
                        interaction_term = input_data[col1].values * input_data[col2].values
                        interaction_terms.append(interaction_term)

        # Convert interaction terms to DataFrame and combine with original input
        if interaction_terms:
            interaction_df = pd.DataFrame(np.column_stack(interaction_terms),
                                           columns=[f"{col1}_x_{col2}" for col1 in input_data.columns for col2 in input_data.columns if col1 != col2])
            input_data = pd.concat([input_data.reset_index(drop=True), interaction_df.reset_index(drop=True)], axis=1)

        # Scale the input data
        scaled_input = self.scaler.transform(input_data)

        # Predict the score differential with the neural network model
        nn_score_differential = self.nn_model.predict(scaled_input)[:, 0]

        # Predict the score differential with the linear regression model
        lr_score_differential = self.linear_model.predict(scaled_input)

        # Average the neural network and linear regression model
        avg_score_differential = (nn_score_differential + lr_score_differential) / 2

        # Use the logistic regression model to convert score differential to win probability
        win_probabilities = self.logistic_model.predict_proba(avg_score_differential.reshape(-1, 1))[:, 1]

        return win_probabilities
