import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

def load_movs_shows(movies, shows):
    """
    Return combined dataset between two given files, one with movies and one
    with tv shows info, keeps all rows and columns from both datasets
    missing values are NaN
    """
    # concat stacks the two datasets, no duplicate column names
    movs_shows = pd.concat([movies, shows], axis=0, sort=True)
    return movs_shows


def load_imdb(imdb, movies):
    """
    Return the joined dataset between two given files, one with imdb and
    one with movie info, returned dataset only has rows where corresponding
    movie titles match in the given two datasets
    Resulting dataset has information of movies present in both given datasets
    """
    merged = movies.merge(imdb, left_on='Title', right_on='title', how='inner')
    return merged


def genre_count(data, site):
    df = data.loc[data[site] == 1, 'Genres']
    df = df.dropna()
    df = df.str.split(',')
    df = df.explode().unique()
    length = len(df)
    print(df)
    print(site, 'has', length, 'genres')


def movs_per_genre(data, site):
    """
    Given dataset and streaming platform, prints total number of movies
    available for each genre in the site
    """
    genres = data.loc[data[site] == 1, data.columns == 'Genres'] 
    genres['Genres'] = genres['Genres'].str.split(",")
    mlb = MultiLabelBinarizer(sparse_output=True)
    df = genres.drop('Genres', 1).join(genres.Genres.str.join('|').str.get_dummies())
    print('The total count of movies per genre for', site)
    print(df.sum(axis=0))


def main():
    movies = pd.read_csv('data/Movies_Streaming_Platforms.csv')
    shows = pd.read_csv('data/TvShows_Streaming_Platforms.csv')
    imdb = pd.read_csv('data/IMDB_movies.csv')
    movs_shows = load_movs_shows(movies, shows)
    imdb_mov = load_imdb(imdb, movies)
    genre_count(movies, 'Netflix')
    genre_count(movies, 'Hulu')
    genre_count(movies, 'Disney+')
    genre_count(movies, 'Prime Video')
    movs_per_genre(imdb_mov, 'Netflix')


if __name__ == '__main__':
    main()
