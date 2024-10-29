import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import json
import os

# Warning suppression
tf.autograph.set_verbosity(10)

# Load YAML configuration
with open('win_prob.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Paths and variables from the config
model_save_folder = config['output']['model_save_folder']
scaler_path = os.path.join(model_save_folder, config['output']['scaler_path'])
nn_model_path = os.path.join(model_save_folder, config['output']['nn_model_path'])
linear_model_path = os.path.join(model_save_folder, config['output']['linear_model_path'])
logistic_model_path = os.path.join(model_save_folder, config['output']['logistic_model_path'])
metrics_save_path = config['output']['metrics_save_path']

# Create the model save folder if it doesn't exist
if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)
    print(f"Created directory: {model_save_folder}")
else:
    print(f"Directory already exists: {model_save_folder}")

# Save model configurations in the model output folder as a JSON file
config_save_path = os.path.join(model_save_folder, "model_config.json")

# Save configuration as a JSON file
with open(config_save_path, 'w') as file:
    json.dump(config, file, indent=4)
print(f"Configuration saved at: {config_save_path}")

# Load your data
train_data = pd.read_csv(config['data']['train_path'], compression='gzip', low_memory=False)
test_data = pd.read_csv(config['data']['test_path'], compression='gzip', low_memory=False)

# Align the training and test datasets
train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)

response_var = config['model']['response_variable']
feature_vars = config['model']['feature_variables']
if config['model']['include_all_features']:
    feature_vars = train_data.drop(columns=[response_var]).columns


X_train = train_data[feature_vars]
y_train = train_data[response_var]
X_test = test_data[feature_vars]
y_test = test_data[response_var]

# Combine X_train and y_train, drop rows with NaN values, then separate
train_data_combined = pd.concat([X_train, y_train], axis=1).dropna()
X_train = train_data_combined[feature_vars]
y_train = train_data_combined[response_var]

# Repeat for the test data
test_data_combined = pd.concat([X_test, y_test], axis=1).dropna()
X_test = test_data_combined[feature_vars]
y_test = test_data_combined[response_var]

# Add interaction terms
interaction_vars = config['model']['interaction_terms']
add_all_interactions = config['model']['include_all_interactions']

interaction_terms = []
if add_all_interactions:
    for col1 in X_train.columns:
        for col2 in X_train.columns:
            if col1 != col2:
                interaction_term = X_train[col1] * X_train[col2]
                interaction_terms.append(interaction_term)
else:
    for col1 in interaction_vars:
        for col2 in interaction_vars:
            if col1 != col2:
                interaction_term = X_train[col1] * X_train[col2]
                interaction_terms.append(interaction_term)

X_train_interactions = pd.DataFrame(np.column_stack(interaction_terms), columns=[f"{col1}_x_{col2}" for col1 in X_train.columns for col2 in X_train.columns if col1 != col2])
X_test_interactions = pd.DataFrame(np.column_stack([X_test[col1] * X_test[col2] for col1 in X_test.columns for col2 in X_test.columns if col1 != col2]), columns=[f"{col1}_x_{col2}" for col1 in X_test.columns for col2 in X_test.columns if col1 != col2])

# Combine original features with interaction terms
X_train_combined = pd.concat([X_train.reset_index(drop=True), X_train_interactions.reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([X_test.reset_index(drop=True), X_test_interactions.reset_index(drop=True)], axis=1)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

# Save the scaler
joblib.dump(scaler, scaler_path)

# Build neural network model
model = keras.Sequential()
for layer in config['model']['nn_architecture']['layers']:
    model.add(layers.Dense(units=layer['units'], activation=layer['activation']))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the neural network
model.fit(X_train_scaled, y_train, epochs=config['training']['epochs'], batch_size=config['training']['batch_size'], validation_split=config['training']['validation_split'])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Save the neural network model
model.save(nn_model_path)

# Predict score differentials on the test set with the neural network
y_pred_nn = model.predict(X_test_scaled).flatten()

# Fit linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Save the linear regression model
with open(linear_model_path, 'wb') as f:
    pickle.dump(linear_model, f)

# Predict score differentials on the test set with linear regression
y_pred_lr = linear_model.predict(X_test_scaled)

# Combine predictions (for example, taking an average)
y_pred_combined = (y_pred_nn + y_pred_lr) / 2

# Convert predictions to binary outcomes based on threshold > 0
y_pred_binary = (y_pred_combined > 0).astype(int)

# Create true binary outcomes based on actual score differentials
y_true = (y_test > 0).astype(int)  # 1 if score differential > 0, 0 otherwise

# Calculate confusion matrix and metrics
cm = confusion_matrix(y_true, y_pred_binary)
accuracy = accuracy_score(y_true, y_pred_binary)
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)

# Print results
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save metrics to JSON
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1
}

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Lose', 'Win'], yticklabels=['Lose', 'Win'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Train logistic regression model to predict win probability based on score_differential predictions
# Reshape y_pred_combined for logistic regression input
y_pred_combined = y_pred_combined.reshape(-1, 1)

# Initialize and train logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(y_pred_combined, y_true)

logistic_model_path = os.path.join(model_save_folder, config['output']['logistic_model_path'])

# Save logistic regression model
with open(logistic_model_path, 'wb') as f:
    pickle.dump(logistic_model, f)

# Predict probability of winning based on score_differential predictions
win_probabilities = logistic_model.predict_proba(y_pred_combined)[:, 1]  # Probability of win (class 1)

# Evaluate the logistic regression model
roc_auc = roc_auc_score(y_true, win_probabilities)
log_loss_val = log_loss(y_true, win_probabilities)

# Print evaluation metrics
print(f"Logistic Regression ROC AUC: {roc_auc:.4f}")
print(f"Logistic Regression Log Loss: {log_loss_val:.4f}")

metrics_save_folder = os.path.dirname(config['output']['metrics_save_path'])
metrics_save_path = config['output']['metrics_save_path']

# Create the model save folder if it doesn't exist
if not os.path.exists(metrics_save_folder):
    os.makedirs(metrics_save_folder)
    print(f"Created directory: {metrics_save_folder}")
else:
    print(f"Directory already exists: {metrics_save_folder}")

# Save metrics to JSON (append to existing metrics)
metrics.update({
    'logistic_roc_auc': roc_auc,
    'logistic_log_loss': log_loss_val
})

with open(metrics_save_path, 'w') as json_file:
    json.dump(metrics, json_file)
print(f"Metrics saved to {metrics_save_path}")

# Print a few sample predictions for verification
print("Sample Win Probabilities:")
for i in range(5):
    print(f"Predicted Score Differential: {y_pred_combined[i][0]:.4f}, Win Probability: {win_probabilities[i]:.4f}")
