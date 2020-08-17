import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def netflix_analysis(df):
    netflix = df[df['Netflix'] == 1]  
    netflix_count = netflix['Genres'].nunique() 
    print(netflix_count)
    netflix_genre = netflix['Genres'].unique()
    netflix['freq'] = netflix.groupby('Genres')['Genres'].transform('count')
    print(netflix)
    print(netflix['freq'])
    genre_list = netflix.groupby('Genres').count()
    print(genre_list)

    #plot = netflix.plot.pie(y='Genres')
    #plt.bar()
    #plt.pie(netflix, labels=netflix_genre, shadow=True)
    #plt.title('Netflix Genre Diversity')
    #plt.show()
    #plt.savefig('charts/netflix_genre_pie_chart.png')


def hulu_analysis(df):
    hulu = df[df['Hulu'] == 1]
    hulu_count = hulu['Genres'].nunique()
    print(hulu_count)
    hulu_genre = hulu['Genres'].unique()
    #plt.pie(hulu, labels=hulu_genre, shadow=True)
    #plt.title('Hulu Genre Diversity')
    #plt.show()
    #plt.savefig('charts/hulu_genre_pie_chart.png')


def disney_analysis(df):
    disney = df[df['Disney+'] == 1]
    disney_count = disney['Genres'].nunique()
    print(disney_count)
    disney_genre = disney['Genres'].unique()
    #plt.pie(disney, labels=disney_genre, shadow=True)
    #plt.title('Disney Genre Diversity')
    #plt.show()
    #plt.savefig('charts/disney_genre_pie_chart.png')


def prime_analysis(df):
    prime = df[df['Prime Video'] == 1]
    prime_count = prime['Genres'].nunique()
    print(prime_count)
    prime_genre = prime['Genres'].unique()
    print(prime_genre)
    #plt.pie(prime, labels=prime_genre, shadow=True)
    #plt.title('Prime Video Genre Diversity')
    #plt.show()
    #plt.savefig('charts/prime_genre_pie_chart.png')

def main():
    movs_shows = load_movs_shows('data/Movies_Streaming_Platforms.csv',
                                 'data/TvShows_Streaming_Platforms.csv')
    df = pd.read_csv('data/Movies_Streaming_Platforms.csv')
    netflix_analysis(df)
    hulu_analysis(df)
    disney_analysis(df)
    prime_analysis(df)


if __name__ == '__main__':
    main()
