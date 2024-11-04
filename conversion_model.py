import pandas as pd
import json
import pickle
import yaml
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import os

# Load configuration
with open("conversion.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract paths and parameters from config
model_save_folder = config['file_paths']['model_save_folder']
data_path = config['file_paths']['data_path']
model_save_path = os.path.join(model_save_folder, config['file_paths']['model_save_path'])
metrics_save_folder = os.path.dirname(config['file_paths']['metrics_save_path'])
metrics_save_path = config['file_paths']['metrics_save_path']
plot_save_path = os.path.join(model_save_folder, config['file_paths']['plot_save_path'])
xgboost_params = config['xgboost_params']
features = config['data']['features']
response = config['data']['response']

# Create the model save folder if it doesn't exist
if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)
    print(f"Created directory: {model_save_folder}")
else:
    print(f"Directory already exists: {model_save_folder}")

# Create metrics save folder if it doesn't exist
if metrics_save_folder and not os.path.exists(metrics_save_folder):
    os.makedirs(metrics_save_folder)
    print(f"Created directory: {metrics_save_folder}")

# Load and preprocess data
df = pd.read_csv(data_path)
X = df[features]
y = df[response]

# Train-test split (test size of 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost model
model = xgb.XGBClassifier(**xgboost_params)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Compute metrics
log_loss_value = log_loss(y_test, y_pred_proba)
r2 = r2_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred.round())

# Save model
with open(model_save_path, 'wb') as file:
    pickle.dump(model, file)

# Save metrics as JSON
metrics = {
    "Log Loss": log_loss_value,
    "R-squared": r2,
    "Accuracy": accuracy
}
with open(metrics_save_path, 'w') as file:
    json.dump(metrics, file, indent=4)

# Generate average actual conversion rates for plotting
avg_actual_conversion = df.groupby('ydstogo')['conv'].mean().reset_index()

# Generate and save plot
plt.figure(figsize=(10, 6))
plt.plot(avg_actual_conversion['ydstogo'], avg_actual_conversion['conv'] * 100, color='blue', label='Average Actual Conversion Rate', marker='o')
plt.scatter(X_test['ydstogo'], y_pred_proba * 100, color='red', label='Predicted Conversion Probability')
plt.xlabel('Yards to Go')
plt.ylabel('Conversion Probability (%)')
plt.title('Predicted vs. Average Actual Conversion Rates Based on Yards to Go')
plt.legend()
plt.grid(True)
plt.savefig(plot_save_path)

print("XGBoost model, metrics (JSON), and plot saved successfully.")
