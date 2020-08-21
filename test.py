"""
Kelly Chu and Khoa Tran
This program tests analysis methods
"""


import analysis
import pandas as pd
from cse163_utils import assert_equals


def test_genre():
    small_movies = pd.read_csv('data/small_movies.csv')
    assert_equals(14, analysis.genre_count(small_movies, 'Netflix'))
    print()
    assert_equals(1, analysis.genre_count(small_movies, 'Hulu'))
    print()
    assert_equals(4, analysis.genre_count(small_movies, 'Disney+'))
    print()
    assert_equals(6, analysis.genre_count(small_movies, 'Prime Video'))
    print()


def test_unique_list():
    small_movies = pd.read_csv('data/small_movies.csv')
    assert_equals(['Action', 'Adventure', 'Sci-Fi', 'Thriller', 'Comedy', 'Western', 'Animation', \
'Family', 'Biography', 'Drama', 'Music', 'War', 'Crime', 'Documentary'], analysis.unquie_list(small_movies, 'Netflix'))
    assert_equals(['Documentary'], analysis.unquie_list(small_movies, 'Hulu'))
    assert_equals(['Action', 'Adventure', 'Drama', 'Family'], analysis.unquie_list(small_movies, 'Disney+'))
    assert_equals(['Western', 'Biography', 'Drama', 'Music', 'War', 'Comedy'], analysis.unquie_list(small_movies, 'Prime Video'))


def test_count():
    small_movies = pd.read_csv('data/small_movies.csv')
    small_tv = pd.read_csv('data/small_tvshows.csv')
    assert_equals(13, analysis.count(small_movies, 'movies', 'Netflix'))
    assert_equals(1, analysis.count(small_movies, 'movies', 'Hulu'))
    assert_equals(1, analysis.count(small_movies, 'movies', 'Disney+'))
    assert_equals(3, analysis.count(small_movies, 'movies', 'Prime Video'))
    assert_equals(13, analysis.count(small_tv, 'shows', 'Netflix'))
    assert_equals(2, analysis.count(small_tv, 'shows', 'Hulu'))
    assert_equals(1, analysis.count(small_tv, 'shows', 'Disney+'))
    assert_equals(1, analysis.count(small_tv, 'shows', 'Prime Video'))


def main():
    """
    Operates test functions
    """
    print()
    test_genre()
    test_count()
    test_unique_list()
    print('Successful')


if __name__ == '__main__':
    main()