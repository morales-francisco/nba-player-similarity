# NBA Player Similarity App

This is an application that tries to find similar NBA players. It takes the stats per game of the 2022-2023 regular season of each player and gets the cosine similarity between each of the them. Twenty-seven variables are taken into account in order to find substitute players with the same players' style of play.

The step-by-step process for its development consisted of the following stages:

- [Scraper](https://github.com/morales-francisco/nba-player-similarity/blob/main/0_scraper.ipynb): collecting statistics, contract and biographical information on NBA players on the Basketball Reference website.
- [Data Cleaning](https://github.com/morales-francisco/nba-player-similarity/blob/main/1_cleaning.ipynb): transformation process to ensure its quality and suitability for analysis, which included checking for null values, outliers, correction of data types and elimination of irrelevant data.
- [Pre Processing](https://github.com/morales-francisco/nba-player-similarity/blob/main/2_pre_processing.ipynb): join the different datasets to create a single large table.
- [Feature Engineering](https://github.com/morales-francisco/nba-player-similarity/blob/main/3_feature_engineering.ipynb): creating new variables.
- [EDA](https://github.com/morales-francisco/nba-player-similarity/blob/main/4_eda.ipynb): exploratory data analysis.
- [Standarization](https://github.com/morales-francisco/nba-player-similarity/blob/main/5_data_normalizaton.ipynb): standardization of the values of the different variables.
- [Model](https://github.com/morales-francisco/nba-player-similarity/blob/main/6_model.ipynb): evaluate the results of the algorithm.
- [Streamlit App](https://github.com/morales-francisco/nba-player-similarity/blob/main/app.py): development of the front end to visualize the results of the algorithm used.
- [Docker](https://github.com/morales-francisco/nba-player-similarity/blob/main/Dockerfile): commands used to dockerize the application
