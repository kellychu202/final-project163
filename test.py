"""
Kelly Chu and Khoa Tran
This program tests analysis methods
"""


import analysis
import pandas as pd
from cse163_utils import assert_equals


def test_genre(data):
    """
    Test the genre count method in analysis
    """
    assert_equals(14, analysis.genre_count(data, 'Netflix'))
    print()
    assert_equals(1, analysis.genre_count(data, 'Hulu'))
    print()
    assert_equals(4, analysis.genre_count(data, 'Disney+'))
    print()
    assert_equals(6, analysis.genre_count(data, 'Prime Video'))
    print()


def test_unique_list(data):
    """
    Test the unqiue_list method in analysis
    """
    assert_equals(['Action', 'Adventure', 'Sci-Fi', 'Thriller', 'Comedy',
                   'Western', 'Animation', 'Family', 'Biography', 'Drama',
                   'Music', 'War', 'Crime', 'Documentary'],
                  analysis.unique_list(data, 'Netflix'))
    assert_equals(['Documentary'], analysis.unique_list(data, 'Hulu'))
    assert_equals(['Action', 'Adventure', 'Drama', 'Family'],
                  analysis.unique_list(data, 'Disney+'))
    assert_equals(['Western', 'Biography', 'Drama', 'Music', 'War', 'Comedy'],
                  analysis.unique_list(data, 'Prime Video'))


def test_count_mov(data):
    """
    Test analysis count method on movie dataset
    """
    assert_equals(13, analysis.count(data, 'movies', 'Netflix'))
    assert_equals(1, analysis.count(data, 'movies', 'Hulu'))
    assert_equals(1, analysis.count(data, 'movies', 'Disney+'))
    assert_equals(3, analysis.count(data, 'movies', 'Prime Video'))


def test_count_tv(data):
    """
    Test analysis count method on tv dataset
    """
    assert_equals(13, analysis.count(data, 'shows', 'Netflix'))
    assert_equals(2, analysis.count(data, 'shows', 'Hulu'))
    assert_equals(1, analysis.count(data, 'shows', 'Disney+'))
    assert_equals(1, analysis.count(data, 'shows', 'Prime Video'))


def test_avg_rating(data):
    """
    Test analysis's avg_rating method
    """
    assert_equals(8.1, analysis.avg_rating(data, 'Disney+'))
    assert_equals(462.1/52, analysis.avg_rating(data, 'Netflix'))


def main():
    """
    Operates test functions
    """
    small_movies = pd.read_csv('data/small_movies.csv')
    small_tv = pd.read_csv('data/small_tvshows.csv')
    small_mov_tv = analysis.load_movs_shows(small_movies, small_tv)
    print()
    test_genre(small_movies)
    test_count_mov(small_movies)
    test_count_tv(small_tv)
    test_unique_list(small_movies)
    test_avg_rating(small_mov_tv)
    print('Successful')


if __name__ == '__main__':
    main()
