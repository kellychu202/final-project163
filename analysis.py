import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
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


def count(data, mov_or_show):
    """
    Prints count of different titles each platform has available in the
    given dataset
    """
    print(mov_or_show, ': In this dataset there are...')
    print(len(data), 'total', mov_or_show, '!')
    netflix = data['Netflix'] == 1
    hulu = data['Hulu'] == 1
    prime = data['Prime Video'] == 1
    disney = data['Disney+'] == 1
    print(len(data[netflix]), 'available on Netflix')
    print(len(data[hulu]), 'available on Hulu')
    print(len(data[prime]), 'available on Prime Video')
    print(len(data[disney]), 'available on Disney+')
    print()


def avg_rating(data):
    """
    Print overall average for each streaming platforms' available
    movie and tv show ratings
    """
    # Convert rotten tomatoes to 1-10 scale to match imdb scale
    avg_rating = data[['Title', 'IMDb', 'Rotten Tomatoes', 'Netflix', 'Hulu',
                       'Prime Video', 'Disney+']]
    avg_rating = avg_rating.dropna()
    print('There are a total of', len(avg_rating),
          'movies and shows with ratings.')
    avg_rating['Rotten Tomatoes'] = avg_rating['Rotten Tomatoes'].str. \
        rstrip('%s')
    avg_rating['Rotten Tomatoes'] = avg_rating['Rotten Tomatoes'].astype(int) \
        / 10
    # average imdb and rotten tomatoe rating for each movie/tv show
    avg_rating['Average Rating'] = (avg_rating['IMDb'] +
                                    avg_rating['Rotten Tomatoes']) / 2
    # dict maps streaming platform to overall average rating
    avg_platf = {}
    netflix = avg_rating['Netflix'] == 1
    hulu = avg_rating['Hulu'] == 1
    prime = avg_rating['Prime Video'] == 1
    disney = avg_rating['Disney+'] == 1
    avg_platf['netflix'] = avg_rating.loc[netflix, 'Average Rating'].mean()
    avg_platf['hulu'] = avg_rating.loc[hulu, 'Average Rating'].mean()
    avg_platf['prime'] = avg_rating.loc[prime, 'Average Rating'].mean()
    avg_platf['disney'] = avg_rating.loc[disney, 'Average Rating'].mean()
    print('Average ratings for available movies and tv shows on each',
          'streaming platform:')
    for platf, rating in avg_platf.items():
        print(platf, ": ", rating)
    print()



def genre_count(data, site):
    """
    Given a dataframe and streaming platform, prints the names and count
    of unique genres available on the platform
    """
    df = data.loc[data[site] == 1, 'Genres']
    df = df.dropna()
    df = df.str.split(',')
    df = df.explode().unique()
    length = len(df)
    print(df)
    print(site, 'has', length, 'genres')


def predict_rating(data):
    """
    train 0.8 0.2 dtr use
    genre turn into list split by "," 
    get_dummies 
    """
    features = data.loc[:, data.columns == 'Genres']
    features['Genres'] = features['Genres'].str.split(",")
    mlb = MultiLabelBinarizer(sparse_output=True)
    df = features.drop('Genres', 1).join(features.Genres.str.join('|').str.get_dummies())
    df.sum(axis=0) 
    

def main():
    # datasets
    movies = pd.read_csv('data/Movies_Streaming_Platforms.csv')
    shows = pd.read_csv('data/TvShows_Streaming_Platforms.csv')
    imdb = pd.read_csv('data/IMDB_movies.csv')
    movs_shows = load_movs_shows(movies, shows)
    imdb_mov = load_imdb(imdb, movies)
    # test
    print(movs_shows.head())
    print(imdb.head())
    # end test
    avg_rating(movs_shows)
    count(movies, 'movies')
    count(shows, 'shows')
    # genre selections
    genre_count(movies, 'Netflix')
    genre_count(movies, 'Hulu')
    genre_count(movies, 'Disney+')
    genre_count(movies, 'Prime Video')


if __name__ == '__main__':
    main()
