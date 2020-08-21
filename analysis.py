"""
Kelly Chu and Khoa Tran
This program examines elements of movies and tv shows from various
streaming platforms 
"""


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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


def count(data, mov_or_show, site):
    """
    Prints count of different titles each platform has available in the
    given dataset
    """
    print(site, ': In this dataset there are...')
    print(len(data), 'total', mov_or_show, '!')
    service = data[data[site] == 1]
    print(len(service), mov_or_show, 'available on', site)
    print()
    return len(service)


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
    avg_rating.loc[avg_rating['Rotten Tomatoes'] < 1, 'Rotten Tomatoes'] = 1
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


def predict_rating(data, feats):
    """
    Given dataframe and column name of features to train on
    Return tuple with mean square error of training
    set and test set

    """
    features = data.loc[:, feats]
    features = features.dropna()
    # one hot encoding genres
    features['Genres'] = features['Genres'].str.split(",")
    temp = features.drop('Genres', 1).join(features.Genres.str.join('|').str.get_dummies())
    labels = temp['IMDb']
    features = temp.loc[:, temp.columns != 'IMDb']
    # ml model
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    train_predictions = model.predict(features_train)
    # training set
    train_error = mean_squared_error(labels_train, train_predictions)
    # test set
    model.fit(features_test, labels_test)
    train_prediction = model.predict(features_test)
    test_error = mean_squared_error(labels_test, train_prediction)
    return (train_error, test_error)


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
    count(movies, 'movies', 'Netflix')
    count(movies, 'movies', 'Hulu')
    count(movies, 'movies', 'Disney+')
    count(movies, 'movies', 'Prime Video')
    count(shows, 'shows', 'Netflix')
    count(shows, 'shows', 'Hulu')
    count(shows, 'shows', 'Disney+')
    count(shows, 'shows', 'Prime Video')
    # genre selections
    genre_count(movies, 'Netflix')
    genre_count(movies, 'Hulu')
    genre_count(movies, 'Disney+')
    genre_count(movies, 'Prime Video')
    print()
    feat = ['Year', 'Genres', 'duration', 'IMDb']
    print('With', feat)
    print('Training set mse', predict_rating(imdb_mov, feat)[0])
    print('Testing set mse', predict_rating(imdb_mov, feat)[1])
    print()
    # ml dtr error with streaming platforms and genres features
    feats1 = ['Netflix', 'Hulu', 'Prime Video',
            'Disney+', 'Genres', 'IMDb']
    print('With', feats1)
    print('Training set mse', predict_rating(imdb_mov, feats1)[0])
    print('Testing set mse', predict_rating(imdb_mov, feats1)[1])
    print()
    # platforms, genres, duration
    feats2 = ['Netflix', 'Hulu', 'Prime Video',
            'Disney+', 'Genres', 'duration', 'IMDb']
    print('With', feats2)
    print('Training set mse', predict_rating(imdb_mov, feats2)[0])
    print('Testing set mse', predict_rating(imdb_mov, feats2)[1])
    print()
    # platforms, genres, year, and duration
    feats3 = ['Year', 'Netflix', 'Hulu', 'Prime Video',
            'Disney+', 'Genres', 'duration', 'IMDb']
    print('With', feats3)
    print('Training set mse', predict_rating(imdb_mov, feats3)[0])
    print('Testing set mse', predict_rating(imdb_mov, feats3)[1])
    print()


if __name__ == '__main__':
    main()
