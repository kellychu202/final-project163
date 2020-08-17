import pandas as pd


def load_movs_shows(path_one, path_two):
    """
    Return combined dataset between two given files, one with movies and one
    with tv shows info, keeps all rows and columns from both datasets
    missing values are NaN
    """
    movies = pd.read_csv(path_one)
    shows = pd.read_csv(path_two)
    # concat stacks the two datasets, no duplicate column names
    movs_shows = pd.concat([movies, shows], axis=0, sort=True)
    return movs_shows


def load_imdb(path_one, path_two):
    """
    Return the joined dataset between two given files, one with imdb and
    one with movie info returned dataset only has rows where corresponding
    movie titles match in the given two datasets
    Resulting dataset has information of movies present in both given datasets
    """
    imdb = pd.read_csv(path_one)
    movies = pd.read_csv(path_two)
    merged = movies.merge(imdb, left_on='Title', right_on='title', how='inner')
    return merged


def main():
    movs_shows = load_movs_shows('data/Movies_Streaming_Platforms.csv',
                                 'data/TvShows_Streaming_Platforms.csv')
    # test
    print(movs_shows.head())
    print(movs_shows.columns)
    print(movs_shows.shape)
    # end test
    imdb = load_imdb('data/IMDB_movies.csv',
                     'data/Movies_Streaming_Platforms.csv')
    # test
    print(imdb.head())
    print(imdb.columns)
    print(imdb.shape)
    # end test


if __name__ == '__main__':
    main()
