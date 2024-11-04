import pandas as pd
import numpy as np

# Load Data
data = pd.read_csv("Data/model_train.csv.gz")

# Filter by 4th down data only
filtered_df = data[data['down_4.0'] == 1]

# Create a DataFrame with selected columns
df = filtered_df[['ydstogo', 'series_result', 'o_rtg', 'opp_d_rtg']].copy()

# Apply the conv column logic with np.select
df['conv'] = np.select(
    [df['series_result'].isin(['First down', 'Touchdown']),
     df['series_result'].isin(['Punt', 'Field goal', 'Missed field goal', 'End of half', 'QB kneel', np.nan])],
    [1, np.nan],
    default=0
)

df = df.dropna(subset=['conv'])

# Output data to a CSV file
df[['ydstogo', 'o_rtg', 'opp_d_rtg', 'conv']].copy().to_csv('Data/conv_rate.csv', index=False)

print("Conversion rate data has been saved to conv_rate.csv")