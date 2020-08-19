import genre
import analysis
import pandas as pd
from cse163_utils import assert_equals

def test_genre():
    small_movies = pd.read_csv('data/small_movies.csv')
    assert_equals(5, genre.genre_count(small_movies, 'Netflix'))
    assert_equals(5, genre.genre_count(small_movies, 'Hulu'))
    assert_equals(5, genre.genre_count(small_movies, 'Disney+'))
    assert_equals(5, genre.genre_count(small_movies, 'Prime Video'))

def test_count():
    small_movies = pd.read_csv('data/small_movies.csv')
    small_tv = pd.read_csv('data/tvshows.csv')
    assert_equals(5, analysis.count(small_movies, 'movies', 'Netflix'))
    assert_equals(5, analysis.count(small_movies, 'movies', 'Hulu'))
    assert_equals(5, analysis.count(small_movies, 'movies', 'Disney+'))
    assert_equals(5, analysis.count(small_movies, 'movies', 'Prime Video'))
    assert_equals(5, analysis.count(small_tv, 'shows', 'Netflix'))
    assert_equals(5, analysis.count(small_tv, 'shows', 'Hulu'))
    assert_equals(5, analysis.count(small_tv, 'shows', 'Disney+'))
    assert_equals(5, analysis.count(small_tv, 'shows', 'Prime Video'))

def main():
    """
    Operates test functions
    """
    test_genre()
    test_count()
    print('Successful')


if __name__ == '__main__':
    main()