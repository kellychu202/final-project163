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


def avg_rating(data):
    """
    Calculate average between rotten tomatoes (converted to 0-10 scale) and
    imdb rating for each movie and tv show and 
    Return (graphical representation) of overall average ratings for each
    streaming platform
    """
    # Convert rotten tomatoes to 1-10 scale to match imdb scale
    avg_rating = data[['Title', 'IMDb', 'Rotten Tomatoes', 'Netflix', 'Hulu', 'Prime Video', 'Disney+']]
    avg_rating = avg_rating.dropna()
    avg_rating['Rotten Tomatoes'] = avg_rating['Rotten Tomatoes'].str.rstrip('%s')
    avg_rating['Rotten Tomatoes'] = avg_rating['Rotten Tomatoes'].astype(int) / 10
    # average imdb and rotten tomatoe rating for each movie/tv show
    avg_rating['Average Rating'] = (avg_rating['IMDb'] + avg_rating['Rotten Tomatoes']) / 2
    # dict maps streaming platform to overall average rating
    avg_platf = {}
    avg_platf['netflix'] = avg_rating.loc[avg_rating['Netflix'] == 1, ['Title', 'Average Rating']].mean()
    avg_platf['hulu'] = avg_rating.loc[avg_rating['Hulu'] == 1, ['Title', 'Average Rating']].mean()
    avg_platf['prime'] = avg_rating.loc[avg_rating['Prime Video'] == 1, ['Title', 'Average Rating']].mean()
    avg_platf['disney'] = avg_rating.loc[avg_rating['Disney+'] == 1, ['Title', 'Average Rating']].mean()
    return avg_platf

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
