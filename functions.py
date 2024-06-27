from parameters import *
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import time
import re
import os
import numpy as np
import warnings
#warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from datetime import datetime


## SAVING DATA ##
def save_raw_data(df,filename, path_raw_data):
    file_path = os.path.join(path_raw_data, filename)
    df.to_csv(file_path, index=False)
    print(f'Data saved in {file_path}')

def save_cleaned_data(df,filename, path_cleaned_data):
    file_path = os.path.join(path_cleaned_data, filename)
    df.to_csv(file_path, index=False)
    print(f'Data saved in {file_path}')

def save_processed_data(df,filename, path_processed_data):
    file_path = os.path.join(path_processed_data, filename)
    df.to_csv(file_path, index=False)
    print(f'Data saved in {file_path}')

def save_normalized_data(df,filename, path_processed_data):
    file_path = os.path.join(path_processed_data, filename)
    df.to_csv(file_path, index=False)
    print(f'Data saved in {file_path}')

def save_model_data(df,filename, path_processed_data):
    file_path = os.path.join(path_processed_data, filename)
    df.to_excel(file_path, index=False)
    print(f'Data saved in {file_path}')

# ------------------------------------------------------------------------------------------------------------------ #

## STAT SCRAPER ##



def get_stats(stat_type):
    global stat_dataframes
    df = pd.DataFrame()
    for season in seasons:
        url = f'https://www.basketball-reference.com/leagues/NBA_{season}_{stat_type}.html'
        html = urlopen(url).read().decode('utf-8')
        html = re.sub(r'(\d+)%', r'\1', html)
        table_html = BeautifulSoup(html, 'html.parser').findAll('table')
        df_season = pd.read_html(str(table_html))[0]
        if stat_type == 'play-by-play':
            df_season.columns = df_season.columns.get_level_values(1)
        else:
            df_season.columns = df_season.columns.to_flat_index()
        df_season = df_season.drop(columns=df_season.columns[0], axis=1)

        df_season['Season'] = season

        df = pd.concat([df_season, df], ignore_index=True)

        print(f'Season {season} - Shape {df_season.shape}')     

        time.sleep(5)
    
    clean_columns_stats(df, stat_type)
    save_raw_data(df, f'nba_{stat_type}_2013_2023.csv', path_raw_data)
    return df

def clean_columns_stats(df,stat_type):
    if 'Unnamed: 19' in df.columns: 
        df.drop(columns=['Unnamed: 19'], inplace=True)
    if 'Unnamed: 24' in df.columns: 
        df.drop(columns=['Unnamed: 24'], inplace=True)
    if stat_type in stat_dataframes:
        df.columns = stat_dataframes[stat_type].columns
    print('Column names - Done')


## CONTRACTS SCRAPER ##
def get_contracts():
    url = 'https://www.basketball-reference.com/contracts/players.html'
    table_html = BeautifulSoup(urlopen(url), 'html.parser').findAll('table')
    df = pd.read_html(str(table_html))[0]
    df.columns = df.columns.to_flat_index() 
    df = df.drop(df.columns[0], axis=1) # drop Rk column
    df = clean_columns_contracts(df)
    save_raw_data(df, 'nba_contracts.csv', path_raw_data)
    return df

def clean_columns_contracts(df):
    columns = ['Player', 'Tm', '2023-24 Salary', '2024-25 Salary', '2025-26 Salary', '2026-27 Salary', '2027-28 Salary', '2028-29 Salary','Guaranteed']
    df.columns = columns
    print('Column names - done')  
    return df


## PLAYERS BIOS SCRAPER##

def get_player_links():
    response = requests.get('https://www.basketball-reference.com/contracts/players.html')
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', class_='sortable')
    player_links = []
    for row in table.find_all('tr'): # Extract the player names and links from the table
        a_element = row.find('a')
        if a_element is not None:
            player_link = a_element.get('href')
        else:
            player_link = None
        player_links.append(player_link)

    player_links = [item for item in player_links if item is not None] 
    return player_links

def get_players_bio():
    player_links = get_player_links()
    player_bio = []
    for player_link in player_links:
        player = {}
        player_url = f'https://www.basketball-reference.com/{player_link}'
        player_rest = requests.get(player_url)
        player_soup = BeautifulSoup(player_rest.content, 'lxml')

        player_info = player_soup.find(name = 'div', attrs = {'id' : 'meta'})

        player['Name'] = player_info.find('span').text.strip()

        string = str(player_info.find_all('p'))

        pattern_position = re.compile('Position:\n  </strong>\n (.*)\n\n')
        player['Position'] = pattern_position.findall(string)

        pattern_shoots = re.compile('Shoots:\n  </strong>\n (.*)\n')
        player['Shoots'] = pattern_shoots.findall(string)

        pattern_height_weight = re.compile('lb</span>\xa0(.*) </p>')
        player['Height-Weight'] = pattern_height_weight.findall(string)

        pattern_data_birth = re.compile('<span data-birth="(.*)" id')
        player['Date Born'] = pattern_data_birth.findall(string)

        pattern_college = re.compile('College:\n    \n    </strong>\n<a(.*)</a>')
        player['College'] = pattern_college.findall(string)

        pattern_draft = re.compile('Draft:\n  </strong>\n<a(.*)')
        player['Draft'] = pattern_draft.findall(string)

        pattern_nba_debut = re.compile('NBA Debut: </strong><a(.*)</a>')
        player['NBA Debut'] = pattern_nba_debut.findall(string)

        pattern_experience = re.compile('Experience:</strong>(.*)\n')
        player['Experience'] = pattern_experience.findall(string)

        player_img = player_soup.find(name='div', attrs={'class': 'media-item'})
        if player_img is not None:
            player['Player Image'] = player_img.find('img')['src']
        else:
            player['Player Image'] = None
            
        player_bio.append(player)
        time.sleep(3)
    df = pd.DataFrame(player_bio)
    print('Data gathering - done')
    save_raw_data(df, f'nba_players_bio.csv', path_raw_data)
    return df




# ------------------------------------------------------------------------------------------------------------------ #

## CLEANING STATS##

def clean_data_stats(stat_type):
    if stat_type in raw_path:
        file_path = raw_path[stat_type]
        df = pd.read_csv(file_path)
        print('Reading df - Done')
    df.drop(df[df['Player'] == 'Player'].index, inplace=True) 
    df.loc[:, 'Player'] = df['Player'].str.replace('*', '', regex=False)
    df.apply(pd.to_numeric, errors='coerce')

    df.drop_duplicates(subset=['Player', 'Season'], keep='first', inplace=True)
    df.sort_values(by='Season', ascending=False, inplace=True)

    
    for column in df.columns:
        df[column] = df[column].apply(lambda x: re.sub(r'^\.', '0.', str(x)))
    df.fillna(0, inplace=True)
    df.replace('nan','0', inplace=True)
   
    if stat_type == 'play-by-play':
        percentage_columns = ['PG%', 'SG%', 'SF%', 'PF%', 'C%']
        df[percentage_columns] = df[percentage_columns].apply(lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce')) / 100

        columns_to_divide = ['BadPass', 'LostBall', 'Fouls Commited Shoot', 'Fouls Commited Off', 'Fouls Drawn Shoot', 'Fouls Drawn Off', 'PGA', 'And1', 'Blkd', 'MP']
        df['G'] = df['G'].astype(float)
        df[columns_to_divide] = df[columns_to_divide].apply(pd.to_numeric, errors='coerce')
        df[columns_to_divide] = round(df[columns_to_divide].div(df['G'], axis=0),2)
    
    print('Data cleaning - Done')
    save_cleaned_data(df, f'nba_{stat_type}_2013_2023_cleaned.csv', path_cleaned_data)
    return df

## CLEANING CONTRACTS ##

def clean_data_contracts(stat_type):
    if stat_type in raw_path:
        file_path = raw_path[stat_type]
        df = pd.read_csv(file_path)
    df.drop_duplicates(subset=['Player'], inplace=True)
    df.drop(df[df['Player'] == 'Player'].index, inplace=True) 
    df.drop(df[df['Player'] == '0'].index, inplace=True) 
    df.drop(df[df['2023-24 Salary'] == 'Salary'].index, inplace=True) 
    df.loc[:, 'Player'] = df['Player'].str.replace('*', '', regex=False)

    df = df.fillna('Free Agent')
    df.replace('nan',0, inplace=True)
    print('Data cleaning - done')
    save_cleaned_data(df, f'nba_{stat_type}_cleaned.csv', path_cleaned_data)
    return df


## CLEANING PLAYERS BIO ##
def clean_data_players_bio(stat_type):
    if stat_type in raw_path:
        file_path = raw_path[stat_type]
        df = pd.read_csv(file_path)
    df['Shoots'] = df['Shoots'].str.extract(r"\['(.*?)'\]")
    df['Position'] = df['Position'].str.extract(r"\['(.*?)'\]")
    
    

    df['Height'] = df['Height-Weight'].str.extract(r'(\d+)cm')
    df['Weight'] = df['Height-Weight'].str.extract(r'0*(\d+)kg')

    df.drop(columns='Height-Weight', inplace=True)

    df['Date Born'] = df['Date Born'].str.extract(r"\['(.*?)'\]")
    df['College'] = df['College'].str.extract(r'">([^"]+)')
    df['College'] = df['College'].str.replace("']", "", regex=False)

    df['Draft Team'] = df['Draft'].str.extract(r'">([^<]+)</a>')

    df['Draft Year'] = df['Draft'].str.extract(r'(\d{4})')

    df['Draft Round'] = df['Draft'].str.extract(r'(\d+.\w+ round)')
    df.drop(columns='Draft', inplace=True)

    df['NBA Debut'] = df['NBA Debut'].str.extract(r'(\w+ \d+, \d{4})')
    df['NBA Debut'] = pd.to_datetime(df['NBA Debut'], format='%B %d, %Y')

    df['Experience'] = pd.to_numeric(df['Experience'].str.extract(r'(\d+)', expand=False), errors='coerce').fillna(0).astype(int)

    fill_values = {
    'College': 'No College',
    'Experience': 0,
    'Draft Team': 'Not Drafted',
    'Draft Year': 'Not Drafted',
    'Draft Round': 'Not Drafted'}
    df.fillna(fill_values, inplace=True)
    df.drop_duplicates(inplace=True)

    print('Data cleaning - done')
    save_cleaned_data(df, f'nba_{stat_type}_cleaned.csv', path_cleaned_data)
    return df

# ------------------------------------------------------------------------------------------------------------------ #

# PRE PROCESSING #
def merge_stats(df1, df2, df3):
    merged_df = df1.merge(df2, on=['Player', 'Season'], how='inner')
    final_merged_df = merged_df.merge(df3, on=['Player', 'Season'], how='inner')
    print('Merge - completed')
    final_merged_df = final_merged_df.loc[:, ~final_merged_df.columns.str.endswith(('_x', '_y'))]
    print('Column names - completed')
    save_processed_data(final_merged_df, 'nba_all_stats.csv', path_processed_data)
    return final_merged_df

def merge_bio_contracts(df1, df2):
    merged_df = pd.merge(df1, df2, left_on='Name', right_on='Player' ,how='left')
    merged_df.drop(columns=['Player', 'Unnamed: 0'], inplace=True)
    merged_df.rename(columns={'Name': 'Player'}, inplace=True)
    print('Merge - completed')
    save_processed_data(merged_df, 'nba_bio_contracts.csv', path_processed_data)
    return merged_df

# ------------------------------------------------------------------------------------------------------------------ #

# FEATURE ENGINEERING #

def shannon_entropy(row):
    entropy = -np.sum(row * np.log(row))
    return entropy

def get_last_season_filtered(df):
    max_season = df['Season'].max()
    df = df[df['Season'] == max_season]
    print(f'All players shape: {df.shape}') 
    percentile_10_G = df['G'].quantile(0.10)
    percentile_10_MP = df['MP'].quantile(0.10)
    df = df[(df['G'] > percentile_10_G) & (df['MP'] > percentile_10_MP)]
    print(f'G and MP above the 10th percentile shape: {df.shape}') 
    save_processed_data(df, 'nba_all_stats_new_features_filtered.csv', path_processed_data)
    return df

def percentiles_df(df):
    percentile_df = pd.DataFrame()
    percentile_df['Player'] = df['Player']
    columns = [col for col in df.columns if col != 'Player']

    for col in columns:
        percentiles = df[col].rank(pct=True)
        percentile_df[col] = percentiles
    reversed_columns = ['TOV', 'PF', 'TOV%', 'BadPass', 'LostBall', 'Fouls Commited Shoot', 'Fouls Commited Off', 'Blkd']
    percentile_df[reversed_columns] = (percentile_df[reversed_columns] - 1) * -1


    print('All stats transformed into percentiles')
    save_processed_data(percentile_df, 'nba_all_stats_new_features_percentiles.csv', path_processed_data)

def df_percentage_difference(df, skip_columns):
    def df_percentage_difference(base, compared):
        return ((compared - base) / base) * 100 if base != 0 else 0

    results = []

    print('Calculating the percentage difference between two players stats')
    
    for base_player in df.index:
        for compared_player in df.index:
            if base_player != compared_player:
                percentage_difference = {
                    'Base player': base_player,
                    'Compared player': compared_player
                }
                for column in df.columns:
                    if column not in skip_columns:
                        base = df.loc[base_player, column]
                        compared = df.loc[compared_player, column]
                        percentage_difference[column] = df_percentage_difference(base, compared)
                results.append(percentage_difference)

    df_final = round(pd.DataFrame(results), 2)
    numerical_columns = df_final.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Convert numeric columns to integers
    df_final[numerical_columns] = df_final[numerical_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    save_processed_data(df_final, 'nba_all_stats_new_features_percentage_difference.csv', path_processed_data)

# ------------------------------------------------------------------------------------------------------------------ #

# EDA #
def plot_histogramm(df, column):
    sns.histplot(df[column])
    plt.show()


def plot_histogram(df, columns):
    num_plots = len(columns)
    num_rows = (num_plots // 3) + (num_plots % 3 > 0)

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    axes = axes.ravel()
    for i, column in enumerate(columns):
        sns.histplot(df[column], ax=axes[i])
        axes[i].set_title(f'{column} histogram')
    
    for i in range(num_plots, num_rows * 3):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def heatmap_correlation(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=['float', 'int']).columns.tolist()
    
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, fmt='.2f', vmin=-1, vmax=1, cmap='coolwarm')
    plt.title('Heatmap Correlation')
    plt.show()
    print("\nCorrelation Table")
    return corr_matrix

def drop_columns(df, columns_to_drop):
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return df

# ------------------------------------------------------------------------------------------------------------------ #

# DATA NORMALIZATION #

def df_normalization(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    reversed_columns = ['TOV', 'PF', 'TOV%', 'BadPass', 'LostBall', 'Fouls Commited Shoot', 'Fouls Commited Off', 'Blkd']
    df[reversed_columns] = df[reversed_columns] * -1

    drop_columns(df, ['Pos', 'Tm'])
    save_normalized_data(df, 'nba_all_stats_normalized.csv', path_normalized_data)
    return df



# ------------------------------------------------------------------------------------------------------------------ #

# STREAMLIT APP  + MODEL#

def get_normalized_stats():
    return pd.read_csv(path_normalized_data + '/nba_all_stats_normalized.csv').set_index('Player')
    
def get_percentage_differences():
    return pd.read_csv(path_processed_data + '/nba_all_stats_new_features_percentage_difference.csv')

def get_percentiles():
    return pd.read_csv(path_processed_data + '/nba_all_stats_new_features_percentiles.csv').set_index('Player')

def get_bio_contracts():
    return pd.read_csv(path_processed_data + '/nba_bio_contracts.csv')

def get_contracts_last_season():
    return pd.read_csv(path_processed_data + '/nba_contracts_last_season.csv')

def get_stats():
    return pd.read_csv(path_processed_data+'/nba_all_stats_new_features_filtered.csv').set_index('Player')

def calculate_age(date_birth):
    date_birth = pd.to_datetime(date_birth)
    reference = datetime(2023, 6, 30)
    age = reference.year - date_birth.year - ((reference.month, reference.day) < (date_birth.month, date_birth.day))
    return age

def map_position(position):
    position_map = {
        'Shooting Guard': 'SG',
        'Center': 'C',
        'Small Forward': 'SF',
        'Power Forward': 'PF',
        'Point Guard': 'PG',
        'Point Guard and Shooting Guard': 'PG-SG',
        'Shooting Guard and Small Forward': 'SG-SF',
        'Center and Power Forward': 'C-PF',
        'Small Forward and Power Forward': 'SF-PF',
        'Power Forward and Center': 'PF-C',
        'Power Forward and Small Forward': 'PF-SF',
        'Shooting Guard and Point Guard': 'SG-PG',
        'Small Forward and Shooting Guard': 'SF-SG',
        'Small Forward,Power Forward, and Shooting Guard': 'SF-PF-SG',
        'Shooting Guard,Small Forward, and Point Guard': 'SG-SF-PG',
        'Small Forward,Shooting Guard, and Power Forward': 'SF-SG-PF',
        'Point Guard,Shooting Guard, and Small Forward': 'PG-SG-SF',
        'Power Forward,Small Forward, and Center': 'PF-SF-C',
        'Power Forward,Small Forward,Point Guard, and Shooting Guard': 'PF-SF-PG-SG',
        'Small Forward,Power Forward,Point Guard,Center, and Shooting Guard': 'SF-PF-PG-C-SG',
        'Shooting Guard,Small Forward, and Power Forward': 'SG-SF-PF',
        'Point Guard,Small Forward, and Shooting Guard': 'PG-SF-SG',
        'Power Forward,Center, and Small Forward': 'PF-C-SF',
        'Small Forward,Power Forward, and Center': 'SF-PF-C',
        'Power Forward,Small Forward, and Shooting Guard': 'PF-SF-SG'
    }
    return position_map.get(position, position)


def top_n_stats(row, n):
    return row.nlargest(n).index.tolist()

def count_common_elements(list1, list2):
    return len(set(list1) & set(list2))

def check_partial_containment(row):
    return row['Pos'] in row['Pos_compared']

def check_salary(row):
    if row['2022-23'] == 0 or row['2022-23_compared'] == 0:
        return True
    else:
        return False
    
def check_salary_difference(row):
    salary = row['2022-23']
    compared_salary = row['2022-23_compared']
    
    if salary == 0 or compared_salary == 0:
        return 0  
    
    diff_percentage = abs(salary - compared_salary) / salary * 100
    return 1 if diff_percentage <= 20 else 0

def check_age_difference(row):
    age = row['Age']
    compared_age = row['Age_compared']
    
    diff_age = abs(age - compared_age)
    return 1 if diff_age <= 2 else 0

def top3_flag(row):
    top3_pct = (row['Top 3 Coincidencies'])/3.0
    return 1 if top3_pct >= 0.66 else 0

def top5_flag(row):
    top3_pct = (row['Top 5 Coincidencies'])/5.0
    return 1 if top3_pct >= 0.66 else 0

def top7_flag(row):
    top3_pct = (row['Top 7 Coincidencies'])/7.0
    return 1 if top3_pct >= 0.66 else 0

def top10_flag(row):
    top3_pct = (row['Top 10 Coincidencies'])/10.0
    return 1 if top3_pct >= 0.66 else 0



def get_player_salary(df, player):
    player_salary = 0
    try:
        player_salary = float(df[df['Player']==player]['2023-24 Salary'].values[0].replace('$', '').replace(',', ''))
    except ValueError:
        player_salary = 0
    return player_salary


def convert_number(salary):
    if salary != 'Free Agent':
        clean_salary = salary.replace('$', '').replace(',', '')
        return int(clean_salary)  # Convert the cleaned string to a float
    else:
        return None  # Return None for 'Free Agent' entries or other non-numeric values

def format_salary(salary):
    if isinstance(salary, str):  
        if salary != 'Free Agent':
            clean_salary = salary.replace('$', '').replace(',', '')
            salary_float = float(clean_salary)
            if salary_float >= 10**6:
                formatted_salary = '${:.2f} M'.format(salary_float / 10**6)
            else:
                formatted_salary = '${:,.0f}'.format(salary_float)
            return formatted_salary
        else:
            return salary
    else:
        return '${:,.0f}'.format(salary) 


def salary_format(value):
    try:
        return "${:,.2f} M".format(float(value)/1000000)
    except ValueError:
        return value 


def find_similar_players(player, columns, threshold, df, df_bios_contracts, df_stats):
    model = NearestNeighbors(metric='cosine')
    
    model.fit(df[columns])

    # Find nearest neighbors and their distances
    nearest_neighbors = model.kneighbors(df.loc[player, columns].values.reshape(1, -1), n_neighbors=447)

    # Obtain indexes and distances
    indexes = nearest_neighbors[1][0][1:]
    distances = nearest_neighbors[0][0][1:]

    #  Calculate similarity and filter by threshold
    similarity_scores = 1 - distances
    result_df = pd.DataFrame({'Player': df.index[indexes], 'Similarity': similarity_scores})
    result_df = result_df[result_df['Similarity'] >= threshold]

    #df_percentage_difference_player = df_percentage_difference[df_percentage_difference['Base player'] == player]
    df_stats = df_stats.reset_index()

    df_stronger_stats = df[columns].apply(lambda row: ', '.join(row.nlargest(10).index), axis=1).reset_index()
    df_stronger_stats.columns = ['Player', 'Top 10 stronger stats']

    df_bios_contracts['Age'] = df_bios_contracts['Date Born'].apply(calculate_age)


    result_df = pd.merge(result_df, df_stats[['Player']+columns], left_on='Player', right_on='Player', how='left')
    result_df = pd.merge(result_df, df_stronger_stats, on='Player', how='left') 
    result_df = pd.merge(result_df, df_bios_contracts[['Player', 'Position', 'Age', '2023-24 Salary']], left_on='Player', right_on='Player', how='left')
    
    stronger_stats = list(df[columns].loc[player].nlargest(10).index) 
    other_columns = list(set(columns) - set(stronger_stats))
    
    result_df = result_df[['Player', 'Position','2023-24 Salary','Age', 'Similarity', 'Top 10 stronger stats'] + stronger_stats + other_columns]
    columns = stronger_stats + other_columns
    
    return result_df
