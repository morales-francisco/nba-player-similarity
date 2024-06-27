import pandas as pd

seasons = list(range(2013,2024))
stat_dataframes = {
    'per_game': pd.DataFrame(columns=['Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                      '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
                                      'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season']),
    'advanced': pd.DataFrame(columns=['Player', 'Pos', 'Age', 'Tm', 'G', 'MP', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%',
                                      'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 
                                      'DBPM', 'BPM', 'VORP', 'Season']),
    'play-by-play': pd.DataFrame(columns=['Player', 'Pos', 'Age', 'Tm', 'G', 'MP', 'PG%', 'SG%', 'SF%', 'PF%', 
                                          'C%', 'OnCourt', 'On-Off', 'BadPass', 'LostBall', 'Fouls Commited Shoot', 
                                          'Fouls Commited Off', 'Fouls Drawn Shoot', 'Fouls Drawn Off', 'PGA', 'And1', 'Blkd', 'Season'])
}
raw_path = {
    'per_game' : 'data/0_raw/nba_per_game_2013_2023.csv',
    'advanced': 'data/0_raw/nba_advanced_2013_2023.csv',
    'play-by-play':'data/0_raw/nba_play-by-play_2013_2023.csv',
    'contracts': 'data/0_raw/nba_contracts.csv',
    'players_bio': 'data/0_raw/nba_players_bio.csv'
}
epsilon = 1e-1000 
path_raw_data = 'data/0_raw'
path_cleaned_data = 'data/1_cleaned'
path_processed_data = 'data/2_processed'
path_normalized_data = 'data/3_normalized'
path_model_data = 'data/4_model'

columns_streamlit = ['FG%','3P%','2P%','eFG%','FT%','ORB','DRB','AST','STL','BLK','TOV','PTS', 'PER', 'OBPM','DBPM','BadPass','LostBall','Fouls Commited Shoot','Fouls Commited Off','Fouls Drawn Shoot','Fouls Drawn Off','PGA','And1','Blkd', '3P Shots Attempted %','2P Shots Attempted %', 'PF Efficiency']

