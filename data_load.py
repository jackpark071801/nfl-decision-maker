import pandas as pd
import numpy as np

def load_and_process_data(year):
    # Load data for the specified year
    url = f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz'
    data = pd.read_csv(url, compression='gzip', low_memory=False)

    # Regular season data only
    data = data.loc[data.season_type=='REG']

    # Calculate game points
    game_points = (
        data.groupby(['game_id', 'posteam', 'defteam', 'posteam_type'])
        .agg(home_score=('home_score', 'max'), away_score=('away_score', 'max'))
        .reset_index()
    )
    
    # Calculate points scored and points allowed
    game_points['points_scored'] = np.where(
        game_points['posteam_type'] == 'home', game_points['home_score'], game_points['away_score']
    )
    game_points['points_allowed'] = np.where(
        game_points['posteam_type'] == 'home', game_points['away_score'], game_points['home_score']
    )

    game_points = game_points[['game_id', 'posteam', 'defteam', 'points_scored', 'points_allowed']].drop_duplicates()
    
    # Calculate offensive and defensive ratings
    season_averages = game_points.groupby('posteam')['points_scored'].mean().reset_index().rename(columns={'posteam': 'team', 'points_scored': 'o_rtg'})
    season_averages_def = game_points.groupby('posteam')['points_allowed'].mean().reset_index().rename(columns={'posteam': 'team', 'points_allowed': 'd_rtg'})
    
    # Ensure consistent data types for merging
    data['posteam'] = data['posteam'].astype(str)
    data['defteam'] = data['defteam'].astype(str)
    season_averages['team'] = season_averages['team'].astype(str)
    season_averages_def['team'] = season_averages_def['team'].astype(str)

    # Join offensive and defensive ratings with main data
    data = data.merge(season_averages, left_on='posteam', right_on='team', how='left').drop(columns=['team'])
    data = data.merge(season_averages_def, left_on='defteam', right_on='team', how='left').drop(columns=['team'])

    # Calculate opposing team's ratings
    data = data.merge(season_averages.rename(columns={'o_rtg': 'opp_o_rtg', 'team': 'defteam'}), left_on='defteam', right_on='defteam', how='left')
    data = data.merge(season_averages_def.rename(columns={'d_rtg': 'opp_d_rtg', 'team': 'defteam'}), left_on='defteam', right_on='defteam', how='left')
    
    # Calculate score differential and filter relevant columns
    data['sc_diff'] = np.where(data['posteam_type'] == 'home', data['home_score'] - data['away_score'], data['away_score'] - data['home_score'])
    
    # One-hot encode the 'down' variable if it exists in your dataset
    if 'down' in data.columns:
        data = pd.get_dummies(data, columns=['down'], prefix='down', drop_first=True)
    
    data.reset_index(drop=True, inplace=True)
    
    return data

# Initialize empty DataFrames for train and test data
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# Process data for each year and save to train or test sets
for year in range(2015, 2025):
    processed_data = load_and_process_data(year)
    if year == 2024:
        # For 2024 data, filter to game_seconds_remaining <= 900 for the test set
        test_data = processed_data[processed_data['game_seconds_remaining'] <= 900]
    else:
        # Append all other years for the train set
        train_data = train_data.append(processed_data, sort=True)
        train_data.reset_index(drop=True, inplace=True)

# Save train and test datasets to compressed CSV
train_data.to_csv('Data/model_train.csv.gz', compression='gzip', index=False)
test_data.to_csv('Data/model_test.csv.gz', compression='gzip', index=False)
